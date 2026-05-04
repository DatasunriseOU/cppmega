# Mamba3 Monolithic CUDA Chunk Wave 29 Lane D Layout Redesign - 2026-04-30

Status: hard no-go
Lane: D
Branch: `worker/mamba3-wave29-layout-redesign`
Base: `worker/mamba3-mono-cuda-chunk` at `51fb94c`

## Scope

Wave29 did not repeat the dead Wave28 all-P/P64 scratch path.  It evaluated
materially different ownership/layout alternatives for the monolithic CUDA
chunk branch:

- Hopper cluster/DSM cross-P reductions between P64 panel CTAs.
- split output ownership with attempted on-chip reuse.
- diagonal-only avoidance of full DK/DQ partials.
- CuTe/Triton hybrid ownership.

The binding baseline remains the measured H200 productionish shape
`B=4, S=4096, H=32, G=1, N=64, P=128, R=4, chunk=16`: Wave8 tile-stream CUDA
computes only `DV`, final `DMIMO_V`, and `DSSDA` in `11.1550079981486 ms`,
while TileLang full `bwd_bwd` is `3.70674 ms`.

## Probe

Added:

```text
upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave29_layout_redesign_gate.py
```

Commands:

```text
python -m py_compile \
  upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave29_layout_redesign_gate.py

python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave29_layout_redesign_gate.py
```

Key output:

- `wave8_subset_ratio_vs_tilelang_full`: `3.009385065623324`
- P64 panel smem: `50688 B` (`49.5 KiB`)
- P64 global partial total from Wave28: `2306867200 B`
  (`2.1484375 GiB`), including `2.0 GiB` for fp32 `DK`/`DQ`.
- P128 recompute-pre-rot smem: `148480 B` (`145.0 KiB`)
- P128 reuse-pre-rot smem: `173056 B` (`169.0 KiB`), `1` active block/SM by
  H100/H200 smem and no fit under local GB10 `99 KiB`.

## Layout Alternatives

### Cluster Cross-P Reduction

Owner: `cluster(B,H,chunk)` with one CTA per P64 panel.

This is the strongest alternative because it can replace the `2.1484375 GiB`
global partial roundtrip with in-kernel cross-P reduction.  For productionish
`P=128`, the cluster has `2` CTAs:

- per-CTA smem: `50688 B`
- cluster smem across two CTAs: `101376 B` (`99.0 KiB`)
- DSM payload for one `DK` or `DQ` phase: `33536 B`
- DSM payload if `DK` and `DQ` are retained together: `66304 B`

No-go reason: this removes the global scratch but does not remove the Wave8
subset work.  It starts from the measured `11.1550079981486 ms` lower bound and
then adds at least two cluster barriers plus DSM peer reads for `DK` and `DQ`.
That is already `3.009385065623324x` over the full TileLang baseline before
adding the missing full-contract consumers.

### Split Output Owner

Owner: separate `DV`/`DMIMO_V`/`DSSDA` P-panel owner plus `DK`/`DQ`/scalar
owner.

No-go reason: CTA-local shared memory cannot be consumed by an unrelated output
owner, so the split owner must either recompute LKQ/state/D or materialize
intermediates globally.  The probe reports:

- extra LKQ `16x16` WMMA ops if recomputed: `32` per logical chunk,
  `1048576` at productionish shape.
- materialized fp32 LKQ traffic: `536870912 B` (`0.5 GiB`).

Both are additions to the `11.155 ms` subset lower bound.

### Diagonal-Only Avoidance

Owner: keep only per-token `R x R` diagonal `qk_dot` and avoid full `DK`/`DQ`
partials.

This is useful as a covered-subset direction, but it is not a full replacement:

- diagonal `qk_dot` bytes: `33554432`
- full fused `qk` bytes: `536870912`
- diagonal/full fraction: `0.0625`
- Wave10 one-launch covered-subset timing: `2.31212 ms`

No-go reason for Lane D: the contract is incomplete because non-diagonal
`DK`/`DQ` terms remain outside this owner.

### CuTe/Triton Hybrid

This is the only direction with a credible full-contract performance story, but
it is not a monolithic CUDA chunk-owner redesign.  It belongs to the existing
TileLang/CuTe/CUTLASS-class production path, with small CUDA helpers only for
covered subsets.

## Local GB10 Compile/Timing

No H200 was used.  A local GB10 component compile/timing run was used only as a
fresh ptxas/resource anchor for the existing compileable Wave8 kernel:

```text
env TORCH_CUDA_ARCH_LIST=12.1 \
  RR_DIAG_CUDA_EXT_SUFFIX=wave29_layout_gb10_smoke \
  RR_DIAG_CUDA_VERBOSE_BUILD=1 \
  python upstream_prs/examples/13_tilelang_floormod_dbz/rr_mono_cuda_chunk_wave8.py \
    --shape smoke --device cuda --iters 5 --warmup 2
```

GPU/runtime:

- GPU: `NVIDIA GB10`
- Torch: `2.13.0.dev20260417+cu132`
- CUDA memory after run: `11.017776489257812 GiB` free /
  `121.62754440307617 GiB` total.
- Shape: smoke, `B=1, S=256, H=4, G=1, N=64, P=64, R=4, chunk=16`

ptxas:

- `stage2_mono_wmma_tile_stream_chunk_owner_kernel`: `56` regs/thread,
  `0` spill stores, `0` spill loads, `1` barrier.
- `reduce_dmimo_chunks_kernel`: `37` regs/thread, `0` spill stores,
  `0` spill loads.

Runtime/resource metadata:

- Dynamic smem: `50692 B`
- Static smem: `0 B`
- Local bytes: `0`
- Active blocks/SM on GB10: `1`
- Theoretical occupancy: `16.666666666666668%`
- Timing samples: `[0.07283200323581696, 0.0716480016708374,
  0.07119999825954437, 0.07056000083684921, 0.0708480030298233] ms`
- Mean timing: `0.07141760140657424 ms`
- Correctness vs bf16-staged torch reference:
  - `DV`: `2.9802322387695312e-08`
  - final `DMIMO_V`: `1.4551915228366852e-10`
  - `DSSDA`: `6.661338147750939e-16`

## Modal Cleanup

Lane D started no Modal app.  `modal app list --json` showed two Wave29
ephemeral apps from other lanes; both had `Tasks=0`, and both were stopped:

- `ap-6fclieiAICxsXr8hknFtpf` -
  `cppmega-mamba3-mono-chunk-wave29-lane-b-d-direct-h100`
- `ap-o3o8f102dndZppsdhga2bA` - `cppmega-wave29-lane-c-h100`

Later `modal app list --json` polling showed new active other-lane apps with
`Tasks=1`:

- `ap-gtGB31s5auw4HvvyjfOCuk` -
  `cppmega-mamba3-mono-chunk-wave29-lane-b-d-direct-memory-h100`
- `ap-5G34NjNGKH8Gs0OZ8AG1FM` - `cppmega-wave29-lane-c-h100`

Lane D did not stop those active apps to avoid interfering with other workers.

## Decision

Do not spend another wave on a full-contract monolithic CUDA layout redesign.

The strongest new candidate, cluster cross-P reduction, only removes the global
partial roundtrip.  It still contains Wave8's already-slower subset as a lower
bound and then adds DSM reduction work for `DK`, `DQ`, and scalars.  Split
ownership loses on-chip reuse, diagonal-only ownership is incomplete, and the
hybrid direction is correctly a TileLang/CuTe/CUTLASS path rather than this
CUDA chunk branch.

The path that deserves work is non-monolithic: keep small CUDA covered-subset
helpers where they are independently fast, and put full state/LKQ/D ownership
in the TileLang/CuTe/CUTLASS-class implementation.
