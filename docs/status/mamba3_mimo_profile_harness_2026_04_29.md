# Mamba3 MIMO Profile Harness - 2026-04-29

Branch: `worker/mamba3-profile-harness`

Base: `worker/mamba3-hopper-tma-ws-fix` (`ccd9679`)

Scope: non-production Modal profiler harness for upstream TileLang Mamba3 MIMO
baseline vs the temp-only Hopper `qk_shared_direct` patch.

## Harness

Added `scripts/modal_mamba3_profile_mimo.py`.

The harness:

- overlays local `cppmega`, `upstream_prs/examples/13_tilelang_floormod_dbz`,
  and `/home/dave/state-spaces-mamba/mamba_ssm` into the Modal container;
- compares `baseline` against `qk_shared_direct` from
  `mamba3_bwd_hopper_tma_ws_fix.patch`;
- records CUDA-event distributions for `bwd_fwd`, `bwd_bwd`, and the chained
  pair;
- emits torch profiler traces/tables with NVTX ranges;
- snapshots TileLang generated sources and source hashes;
- writes `report.json`, `elapsed_samples.csv`, torch traces, profiler tables,
  and generated kernel sources to Modal Volume `cppmega-mamba3-profiles`.

Run command used:

```bash
CPPMEGA_MODAL_GPU=H200:2 \
CPPMEGA_MAMBA3_PROFILE_SHAPES=small,prodish \
CPPMEGA_MAMBA3_PROFILE_ITERS=20 \
CPPMEGA_MAMBA3_PROFILE_WARMUP=3 \
CPPMEGA_MAMBA3_TORCH_PROFILE=1 \
modal run scripts/modal_mamba3_profile_mimo.py
```

## Modal Runs

| App | Device | Result | Volume path |
| --- | --- | --- | --- |
| `ap-av2unpZo5wlKLJkzIFarWT` | H200 | failed after compile/source snapshot due harness `segsum` dict bug | `/mamba3_mimo_profile/20260429_105906` |
| `ap-QGnpIC8Ot19ZFEReHPZKMy` | H200 | successful small + prodish A/B profile | `/mamba3_mimo_profile/20260429_110500` |

Successful run metadata:

- actual device: `NVIDIA H200`, capability `(9, 0)`, `device_count=2`;
- requested GPU: `H200:2`;
- torch: `2.13.0.dev20260426+cu132`, CUDA `13.2`;
- TileLang: `0.1.8+cu132.gitf309d814`;
- `ncu`: `/usr/local/cuda/bin/ncu`, Nsight Compute `2026.1.1.0`;
- `nsys`: `/usr/local/bin/nsys`, Nsight Systems `2026.1.1.0`.

Primary artifacts:

- `cppmega-mamba3-profiles:/mamba3_mimo_profile/20260429_110500/report.json`
- `cppmega-mamba3-profiles:/mamba3_mimo_profile/20260429_110500/elapsed_samples.csv`
- per-shape/per-variant:
  - `bwd_fwd_kernel_source.cu`
  - `bwd_bwd_kernel_source.cu`
  - `<shape>_<variant>_torch_trace.json`
  - `<shape>_<variant>_torch_cuda_table.txt`

Example retrieval:

```bash
modal volume get cppmega-mamba3-profiles /mamba3_mimo_profile/20260429_110500 ./mamba3_mimo_profile_20260429_110500
```

## CUDA Event Results

20 measured samples per phase, after 3 warmup iterations.

| Shape | Variant | bwd_fwd mean / p99 ms | bwd_bwd mean / p99 ms | chain mean / p99 ms |
| --- | --- | ---: | ---: | ---: |
| `small` B1 S256 H4 N64 P64 R4 | baseline | 0.0793 / 0.0840 | 0.1649 / 0.1700 | 0.2290 / 0.2330 |
| `small` | qk_shared_direct | 0.0818 / 0.0863 | 0.1652 / 0.1711 | 0.2305 / 0.2349 |
| `prodish` B2 S2048 H16 N64 P64 R4 | baseline | 0.6824 / 0.6868 | 1.3046 / 1.3100 | 1.9100 / 1.9193 |
| `prodish` | qk_shared_direct | 0.7335 / 0.7399 | 1.3007 / 1.3061 | 1.9518 / 1.9579 |

Speed ratio is `baseline_mean / qk_shared_direct_mean`:

- `small`: `bwd_fwd=0.970`, `bwd_bwd=0.998`, `chain=0.993`;
- `prodish`: `bwd_fwd=0.930`, `bwd_bwd=1.003`, `chain=0.979`.

Output absmax summaries matched between baseline and qk_shared_direct for both
shapes in this deterministic harness run.

## TileLang Source Metadata

Generated-source snapshots were written for both kernels and variants.

Small shape:

- baseline `bwd_fwd`: 39,464 chars, 589 lines,
  sha256 `cca24393cb8a30aea1d91b2df488c1f087ff9f0c80234388dec53a3aab8bf2ba`
- baseline `bwd_bwd`: 89,315 chars, 1,184 lines,
  sha256 `8fefdf88af93b06c9753e9e4fa24dbe001a74489d833ff216c49244a3399244d`
- qk `bwd_fwd`: 39,639 chars, 596 lines,
  sha256 `76959f68dc9ceee88958cf0a3d23ad911629e4267450b755fc40b854f16f6af1`
- qk `bwd_bwd`: 89,245 chars, 1,179 lines,
  sha256 `5a07c2d8602644c6243733c6792477421c543fcc83a8ee82ee25a35647a09a2a`

Prodish shape:

- baseline `bwd_fwd`: 40,039 chars, 589 lines,
  sha256 `790975c2a24dd223cbec5896d1ceb0088956e06bc87714f6608adac043ae974f`
- baseline `bwd_bwd`: 90,431 chars, 1,184 lines,
  sha256 `800126161c58780e3369c65a16625fe3b6af83dd09f0eb1c0c25983082b8e0bd`
- qk `bwd_fwd`: 40,199 chars, 596 lines,
  sha256 `655362bdb7d1fc599fa666b8712e916073e6eb3fe7a6908a6c8d33b660efbed6`
- qk `bwd_bwd`: 90,361 chars, 1,179 lines,
  sha256 `0295b5af5e555cdf4a5527be5a6cee1e75ea4cd58abdfe14c38a1fbdc5b3dbc0`

## Bottleneck Readout

- `bwd_bwd` dominates elapsed time: about 72% of `small` chain and 67% of
  `prodish` chain.
- `qk_shared_direct` is effectively neutral on `bwd_bwd`: `small` is 0.2%
  slower, `prodish` is 0.3% faster.
- The chain is slower for `qk_shared_direct` because `bwd_fwd` regresses:
  about 3.1% on `small` and 7.5% on `prodish`.
- TileLang still prints `[WS] skipped: no TMA copies in pipeline loop`, so
  enabling Hopper TMA/WS pass config does not produce a producer-consumer
  warp-specialized pipeline for these loops.
- TileLang repeatedly reports swizzle layout conflicts for `dstates_shared`,
  `dPhiO_shared`, `q_shared`, and `k_shared`, which is the most visible compile
  signal pointing at layout pressure.

Current conclusion: `qk_shared_direct` is a valid compile/smoke workaround for
the Hopper TMA path, but this H200 profile does not show an elapsed-time win.
It only removes the fragile local-fragment qk copy in `bwd_bwd`; the measured
end-to-end chain is gated by `bwd_bwd` itself plus a `bwd_fwd` regression.
