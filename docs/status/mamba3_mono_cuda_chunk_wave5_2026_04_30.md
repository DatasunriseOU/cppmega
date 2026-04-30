# Mamba3 Monolithic CUDA Chunk Wave 5 - 2026-04-30

Branch: `worker/mamba3-mono-cuda-chunk`

## Goal

Stop chasing the Wave 2-4 chunk-owner WMMA fallback and build the first
scan-owner CUDA skeleton for the monolithic Mamba3 backward chunk idea.

Target schedule:

- one CTA owns one `(B, H)` scan;
- reverse-loop over chunks inside that CTA;
- compute `LKQ = K @ Q.T` once per chunk;
- reuse the live `LKQ` / masked-`LKQ` state across `DV`, `DMIMO_V`, and
  `DSSDA` consumers;
- keep final `DMIMO_V` local to the CTA where feasible.

## Implemented

Files:

- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5_extension.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5_kernel.cu`
- `scripts/modal_mamba3_mono_cuda_chunk_wave5.py`
- `docs/status/mamba3_mono_cuda_chunk_wave5_2026_04_30.md`

The prototype keeps the Wave 2 bf16-staged WMMA math contract so it can compare
directly against the existing torch reference.  It changes only ownership and
output shape:

- Wave 2-4 returned per-chunk `DMIMO_V` partials.
- Wave 5 returns final `DMIMO_V[B,H,R,P]`, accumulated in CTA-local shared
  memory across the reverse chunk loop.

This wave intentionally uses the full masked tile for `masked(LKQ) @ dPhi`.
Triangular causal tile-k pruning is deferred until after the scan-owner contract
is stable.

## Resource Shape

Productionish shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.

| metric | Wave 4 P64 panel owner | Wave 5 scan owner |
| --- | ---: | ---: |
| owner | `(B,H,chunk,P64-panel)` | `(B,H)` |
| CTAs | `65536` | `128` |
| chunks per CTA | `1` | `256` |
| logical chunk visits | `32768` | `32768` |
| P64 panels per logical chunk | `2` | `2` |
| dynamic smem | `65540 B` | `68612 B` |
| registers/thread on H200 | `40` | `190` |
| active blocks/SM on H200 | `3` | `1` |
| theoretical occupancy on H200 | `37.5%` | `12.5%` |
| final `DMIMO_V` shared bytes | n/a | `2048 B` |
| `DSSDA` shared bytes | n/a | `1024 B` |

The intended positive change is that `LKQ` is computed once per logical chunk
and reused across both P64 panels.  The immediate blocker is much larger: the
production shape has only `B*H = 128` CTAs, less than one CTA per H200 SM.

## Local Checks

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5.py \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5_extension.py \
  scripts/modal_mamba3_mono_cuda_chunk_wave5.py

git diff --check
```

GB10 smoke compile/correctness:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave5_scan_owner \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5.py \
  --shape smoke --device cuda --iters 1 --warmup 0
```

GB10 smoke results:

| comparison | `DV` | final `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `2.9802322387695312e-08` | `1.7462298274040222e-10` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `9.119758033193648e-07` | `9.892642260922457e-12` |

GB10 smoke metadata:

| metric | value |
| --- | ---: |
| registers/thread | `80` |
| dynamic smem | `67588 B` |
| active blocks/SM | `1` |
| theoretical occupancy | `16.67%` |

GB10 P128 two-panel check:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=local_gb10_wave5_scan_owner \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave5.py \
  --B 1 --S 256 --H 4 --G 1 --N 64 --P 128 --R 4 --chunk 16 \
  --device cuda --iters 1 --warmup 0
```

| comparison | `DV` | final `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `2.384185791015625e-07` | `1.7462298274040222e-10` | `1.3322676295501878e-15` |
| vs Wave 1 fp32 ref | `9.5367431640625e-07` | `1.110718585550785e-06` | `1.368505309073953e-11` |

GB10 P128 one-sample timing: `2.9996800422668457 ms`.

## H200 Smoke And Productionish

Smoke command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave5.py::run_remote \
  --shape-csv smoke \
  --warmup 1 \
  --iters 3
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Modal app: `ap-7ke3M230DrV09MYPR0czsF`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- App stopped after local entrypoint completed.

H200 smoke correctness:

| comparison | `DV` | final `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `1.862645149230957e-09` | `1.4551915228366852e-10` | `6.661338147750939e-16` |
| vs Wave 1 fp32 ref | `4.76837158203125e-07` | `9.541254257783294e-07` | `9.892642260922457e-12` |

H200 smoke timing:

| path | mean ms | min ms | p50 ms | max ms |
| --- | ---: | ---: | ---: | ---: |
| Wave 5 scan-owner slice | `0.4539199968179067` | `0.45151999592781067` | `0.4525440037250519` | `0.45769599080085754` |

Productionish one-sample command:

```text
timeout 1800s modal run --timestamps \
  scripts/modal_mamba3_mono_cuda_chunk_wave5.py::run_remote \
  --shape-csv productionish \
  --warmup 0 \
  --iters 1
```

Runtime:

- GPU: `NVIDIA H200`
- Torch: `2.13.0.dev20260426+cu132`
- Modal app: `ap-0hNvO9jVO5y9g6iHzlWKdS`
- Image: `ghcr.io/jewelmusicee/cppmega:785c3fd`
- App stopped after local entrypoint completed.

H200 productionish correctness:

| comparison | `DV` | final `DMIMO_V` | `DSSDA` |
| --- | ---: | ---: | ---: |
| vs bf16-staged torch ref | `4.76837158203125e-07` | `2.561137080192566e-09` | `2.6645352591003757e-15` |
| vs Wave 1 fp32 ref | `9.5367431640625e-07` | `5.432055331766605e-06` | `3.33120198092729e-11` |

H200 productionish one-sample timing:

| path | mean ms | ratio vs TileLang full `bwd_bwd` | delta vs Wave 4 |
| --- | ---: | ---: | ---: |
| Wave 5 scan-owner slice | `14.08131217956543` | `3.798839999451116x` | `+5.296960258483887 ms` |

## Verdict

Wave 5 is a useful correctness/resource skeleton, not a performance path.

It proves the scan-owner contract can reuse chunk-local `LKQ` across P panels
and accumulate final `DMIMO_V` locally while staying within a GB10/H200 smem
budget.  The blocker is exposed clearly: one CTA per `(B,H)` gives only `128`
CTAs for the production shape, so the kernel underfills H200 before math
efficiency matters.  The next design step needs a split scan ownership model,
for example chunk-group or head-panel scan CTAs with a deliberate final state /
`DMIMO_V` reduction plan, rather than more WMMA tuning inside this exact owner.
