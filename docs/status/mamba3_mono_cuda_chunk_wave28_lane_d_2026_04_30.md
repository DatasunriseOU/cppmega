# Mamba3 Monolithic CUDA Chunk Wave 28 Lane D - 2026-04-30

Status: hard no-go
Lane: D
Branch: `worker/mamba3-mono-cuda-chunk`

## Scope

Lane D re-opened the monolithic CUDA chunk question with the specific constraint
that one ownership/layout should reuse LKQ/state intermediates across:

- `DV`
- `DMIMO_V`
- `DK` / `DQ`
- scalar consumers including `DGAMMA_DIAG`, `DDA`, `DDA_CS`, `DDA_CS_REV`,
  `DFACTOR`, and `DSSDA`

The prior evidence was treated as binding: the best final-output monolithic CUDA
subset was Wave 8 at about `11.15 ms` on H200, while the TileLang full
`bwd_bwd` baseline is `3.70674 ms`.

## Budget

No H200 was used for this lane.  The only GPU run in this pass was a local
GB10 component sanity/compile run.  This follows the budget rule: mini/component
tests use H100 or local GPU only; H200 is reserved for full-size 20-step runs.

Modal app state was checked with:

```text
modal app list --json
```

The visible wave apps were stopped with `0` tasks.  Lane D started no Modal app.

## Internal Iterations

1. Baseline check: Wave 8 tile-stream WMMA keeps tensor-core LKQ/state work and
   removes full LKQ residency, but only covers `DV`, final `DMIMO_V`, and
   `DSSDA`.
2. P64 all-consumer owner without reductions: rejected because `DK`/`DQ` and
   scalar consumers need sums over the full P dimension.
3. P64 owner with global partials: rejected because productionish fp32 partials
   require about `2.148 GiB` of roundtrip scratch before reduction.
4. P128 all-P owner: avoids the P reduction but needs roughly `169 KiB` dynamic
   shared memory when retaining pre-rot/pre-trap tensors for scalar consumers.
5. P128 recompute-pre-rot variant: lowers modeled smem to roughly `145 KiB`,
   but spends extra scalar/rotary work in a path already slower than TileLang.
6. Full LKQ residency: rejected by prior Wave 5/6 timings (`14.08-14.52 ms`)
   and one-block/SM behavior.
7. Row-stream LKQ: rejected by prior Wave 7 timing (`179.77 ms`), proving scalar
   LKQ streaming destroys throughput.
8. Output-owner `DMIMO_V`: kept as evidence for the non-monolithic covered
   subset, but it is not a single chunk-owner layout.
9. One-launch covered subset: Wave 10 shows the covered diagonal/qk subset can
   run at `2.31212 ms`, but state/LKQ/D and non-diagonal `DK`/`DQ` remain
   outside it.
10. Decision: no full monolithic CUDA owner has a credible path below the
    `3.70674 ms` TileLang full baseline while preserving LKQ/state reuse and
    avoiding global roundtrips.

## Resource Gate

Added:

```text
upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_lane_d_resource_gate.py
```

Command:

```text
python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_lane_d_resource_gate.py
```

Productionish model:

- Shape: `B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`.
- P64 panel owner smem: `50688 B`.
- P64 all-consumer global scratch if retained: `2.1484375 GiB` total, including
  `2.0 GiB` for fp32 `DK`/`DQ` partials.
- P128 all-P owner, recompute pre-rot/pre-trap: `148480 B` (`145.0 KiB`).
- P128 all-P owner, retain pre-rot/pre-trap for scalar consumers: `173056 B`
  (`169.0 KiB`).
- P128 all-P owner fits H100/H200 dynamic smem but only at `1` active block/SM
  by smem; it does not fit the local GB10 `99 KiB` limit.

## Local GB10 Compile/Timing

Command:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=lane_d_wave28_gb10_baseline \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
    --shape smoke --device cuda --iters 3 --warmup 1
```

GPU/runtime:

- GPU: `NVIDIA GB10`
- Torch: `2.13.0.dev20260417+cu132`
- Local CUDA memory info after the run: `16.3035 GiB` free /
  `121.6275 GiB` total.
- Shape: smoke, `B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`

ptxas:

- `stage2_mono_wmma_tile_stream_chunk_owner_kernel`: `56` regs/thread,
  `0` spill stores, `0` spill loads, `1` barrier.
- `reduce_dmimo_chunks_kernel`: `37` regs/thread, `0` spill stores,
  `0` spill loads.

Runtime/resource metadata:

- Dynamic smem: `50692 B`
- Active blocks/SM on GB10: `1`
- Theoretical occupancy: `16.6667%`
- Timing samples: `0.073344`, `0.070176`, `0.070208 ms`
- Mean timing: `0.071242665 ms`
- Correctness vs bf16-staged torch reference:
  - `DV`: `2.9802322387695312e-08`
  - final `DMIMO_V`: `1.4551915228366852e-10`
  - `DSSDA`: `6.661338147750939e-16`

## Decision

Do not continue the monolithic CUDA chunk-kernel path for full Mamba3
`bwd_bwd`.

The reason is structural, not just tuning: the P64 owner is the only low-live-set
tensorized layout, but full consumers force a global P-panel roundtrip.  The
P128 owner removes that roundtrip but becomes a one-block/SM all-consumer CTA
and still adds substantial missing `DK`/`DQ`/scalar work to a measured subset
that was already about `3x` slower than TileLang full `bwd_bwd`.

The path that deserves more work is the non-monolithic covered-subset direction:
keep the fast output-owner `DMIMO_V`/diagonal CUDA pieces separate, or move the
full state/LKQ/D work to a TileLang/CuTe/CUTLASS-class WGMMA design rather than
expanding the monolithic CUDA WMMA owner.
