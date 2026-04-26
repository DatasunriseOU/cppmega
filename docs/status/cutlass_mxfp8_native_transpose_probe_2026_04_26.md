# CUTLASS MXFP8 Native Transpose Probe

Date: 2026-04-26

Scope:

- `tools/probes/cutlass_mxfp8_native_transpose_probe.cu`
- `tools/probes/cutlass_mxfp8_native_transpose_probe.py`

## Result

The native CUTLASS/CuTe SM121 MXFP8 block-scaled GEMM path is buildable and
runs on GB10 when compiled with an accelerated target (`sm_121a`). A small
128x128x128 TN GEMM with A/B MXFP8 E4M3, BF16 D, and native SM1xx scale layouts
passes CUTLASS host reference verification.

Directly pointing the CUTLASS builder path at TE compact columnwise scales for a
logical transpose is not correct. The storage size matches, but the byte order
does not:

```text
native_vs_te_rowwise_mismatches=504 / 512
native_vs_te_columnwise_transpose_alias_mismatches=510 / 512
cute_layout_vs_te_columnwise_transpose_alias_mismatches=0 / 512
```

The payload transpose is plausibly expressible as an ordinary dense operand
layout: TE source row-major `[orig_rows, orig_cols]` can be read as the logical
transpose through a column-major CUTLASS operand view. The blocker is scale
layout, not payload layout. A CuTe layout can represent the TE
columnwise-as-transpose scale byte mapping, but the stock CUTLASS builder does
not expose that layout as `LayoutSFA/LayoutSFB`.

## Source Points

- CUTLASS example 79c uses `OpClassBlockScaledTensorOp`, `Sm120`, and obtains
  `LayoutSFA/LayoutSFB` from the generated mainloop:
  `/home/dave/vllm/.deps/cutlass-src/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu:134`
  and `:155`.
- The same example materializes native scale layouts with
  `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB` before allocating scale
  storage:
  `/home/dave/vllm/.deps/cutlass-src/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu:344`.
- `Sm1xxBlockScaledBasicChunk<32>` defines the native scale atom as a swizzled
  128x128 element tile. For K-major input scales, offsets follow the atom
  strides in
  `/home/dave/vllm/.deps/cutlass-src/include/cutlass/detail/sm100_blockscaled_layout.hpp:54`.
- `tile_atom_to_shape_SFA/SFB` maps SFA over `(M,K)` and SFB over `(N,K)`:
  `/home/dave/vllm/.deps/cutlass-src/include/cutlass/detail/sm100_blockscaled_layout.hpp:86`.
- The SM120 builder fixes `LayoutSFA/LayoutSFB` to
  `Sm1xxBlockScaledConfig::deduce_layoutSFA/SFB`; there is no public builder
  knob for a TE compact scale layout:
  `/home/dave/vllm/.deps/cutlass-src/include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl:216`.
- The mainloop TMA path does consume the layout object when creating scale TMA
  copies:
  `/home/dave/vllm/.deps/cutlass-src/include/cutlass/gemm/collective/sm120_blockscaled_mma_tma.hpp:298`
  and `:306`.
- The local probe checks the stock native scale layout, the TE compact formulas,
  and a CuTe layout that exactly matches TE columnwise-as-transpose offsets:
  `tools/probes/cutlass_mxfp8_native_transpose_probe.cu:112`.
- Existing TE no-copy probes already showed PyTorch `.t()` metadata is not
  enough for TE `general_gemm`:
  `docs/te_mxfp8_backward_gb10_plan_2026_04_25.md:29` and `:86`.

## Commands

```bash
python tools/probes/cutlass_mxfp8_native_transpose_probe.py \
  --layout-only \
  --build-dir /tmp/cppmega_cutlass_mxfp8_probe_build
```

Result: build passed; layout mismatch counts above.

```bash
python tools/probes/cutlass_mxfp8_native_transpose_probe.py \
  --build-dir /tmp/cppmega_cutlass_mxfp8_probe_build \
  --m 128 --n 128 --k 128
```

Result:

```text
device
  name=NVIDIA GB10
  compute_capability=12.1
native_gemm
  problem=128x128x128
  layout=TN A=row-major B=column-major
  scale_layout=native_sm1xx_blockscaled
  disposition=passed
```

Control run with plain nvcc `-arch=sm_121` compiled, but kernel launch hit CUTLASS'
device-side assert: `ERROR : Arch conditional MMA instruction used without
targeting appropriate compute capability`. Use `sm_121a` for this probe.

## Recommendation

Do not try to remove the transpose by passing TE compact `columnwise_scale_inv`
into the stock CUTLASS builder path. It will read the wrong scale bytes for
almost every scale slot.

The next useful experiment is a narrow CUTLASS mainloop/builder fork or manual
`CollectiveMma` instantiation that keeps the native SM1xx shared-memory scale
layout but supplies a custom CuTe global-memory scale layout:

```text
logical scale for transpose: scale(row=orig_col, k=orig_row)
TE columnwise byte offset:  (orig_row / 32) * orig_cols + orig_col
```

The local probe proves this mapping is representable as a CuTe layout. The
remaining unknown is whether the CUTLASS TMA path accepts it when supplied
through a manual/forked mainloop type. If yes, this avoids global materialized
scale transpose and lets TMA repack scales into the native SM1xx shared-memory
layout. If TMA rejects it, the remaining options are a small global scale
prepack or a deeper mainloop fork with custom scale loads.

## Update: Manual Mainloop Attempts

Files added/updated:

- `tools/probes/cutlass_mxfp8_native_transpose_probe.cu`
- `tools/probes/cutlass_mxfp8_native_transpose_probe.py`
- `tools/probes/cutlass_mxfp8_te_compact_gemm_probe.py`

The stock native GEMM remains buildable and passing on GB10. The direct
custom-global-scale `CollectiveMma` instantiation also compiles, but the runtime
TMA descriptor path rejects the compact TE scale layout:

```text
copy_traits_sm90_tma.hpp:977:
Assertion `(gmem_prob_stride[1] & 0b1111) == 0' failed.
```

For larger `K=512`, that stride assert is avoided, but CUDA tensor-map
descriptor initialization still fails for the compact zero-stride scale box:

```text
Error: Failed to initialize the TMA descriptor 1
globalDim      (8,128,1,1,1)
globalStrides  (2,16,0,0,0)
boxDim         (2,1,1,1,1)
```

A second Python-callable probe tried the other half of the problem: keep scales
prepacked to native SM1xx layout, but read the original TE columnwise payload
`[K,N]` as logical B without materializing `[N,K]`. The manual B stride
`stride_N=1,stride_K=N` reaches TMA descriptor creation and aborts with:

```text
copy_traits_sm90_tma.hpp:968:
Assertion `gmem_prob_stride[0] == 1 && "Majorness of smem doesn't match majorness of gmem"' failed.
```

This also corrected an earlier assumption: the stock SM120 `ColumnMajor` B path
does not alias original `[K,N]` TE columnwise payload as desired. The probe's
output matched `A @ B.T`, meaning the stock path expects a physically
materialized `[N,K]` payload for the backward transpose case.

The runnable default probe now keeps aborting experiments opt-in:

```bash
python tools/probes/cutlass_mxfp8_native_transpose_probe.py \
  --build-dir /tmp/cppmega_cutlass_mxfp8_probe_build \
  --m 128 --n 128 --k 128
```

Result:

```text
native_gemm
  disposition=passed
te_compact_gemm
  disposition=not_run
  reason=direct compact scale TMA attempt aborts on current CUTLASS/CUDA; pass --attempt-te-compact to reproduce
```

Opt-in repros:

```bash
python tools/probes/cutlass_mxfp8_native_transpose_probe.py \
  --build-dir /tmp/cppmega_cutlass_mxfp8_probe_build \
  --m 128 --n 128 --k 128 --attempt-te-compact
```

Expected result: process aborts in scale TMA descriptor creation.

```bash
python tools/probes/cutlass_mxfp8_te_compact_gemm_probe.py \
  --build-dir /tmp/cppmega_cutlass_mxfp8_te_compact_build \
  --m 128 --n 128 --k 128 --attempt-no-copy-payload
```

Expected result: process aborts in B payload TMA descriptor creation.

Current conclusion: a production GB10 no-materialized-transpose path needs a
deeper SM120 mainloop fork that bypasses TMA for at least the transposed payload
and compact scales, or a small custom loader that gathers TE compact/global
layouts into the native shared-memory layout. A Python/shim integration was not
added because the native no-copy CUTLASS path is not callable without hitting
these descriptor assertions.
