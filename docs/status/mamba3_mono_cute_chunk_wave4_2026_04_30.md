# Mamba3 Mono CuTe Chunk Wave 4 - 2026-04-30

Branch: `worker/mamba3-mono-cute-chunk`

## Goal

Make the hand-written CuTe DSL `64x64x64` BF16 WGMMA GEMM numerically correct
on Modal H200. Performance is secondary for this wave.

## Files Changed

- `cppmega/megatron/cute_dsl_mimo/single_gemm_test.py`
- `docs/status/mamba3_mono_cute_chunk_wave4_2026_04_30.md`

## Diagnosis

Wave 3 fixed the CuTe DSL vectorized-copy verifier failure, but the
hand-written kernel was still numerically wrong:

- `max_abs`: `17.318359`
- `max_rel`: `0.615764`

Wave 4 added deterministic cases before timing:

- `identity_transpose`: catches B transposition / basic C store mapping.
- `structured_mod`: catches K-lane and swizzled-copy corruption.
- `random_seed_42`: preserves the original seeded random smoke.

The key correctness bug was the `128`-bit universal copy between row-major gmem
and swizzled smem. It compiled after the Wave 3 2D tiler change, but it still
copied groups across swizzled shared-memory layout boundaries and corrupted
multi-K dot products. The identity case passed, while `structured_mod` still
failed with repeated tail-column errors such as:

```text
Case structured_mod: max_abs=6.517822 max_rel=1.568198
top errors at columns 59/61 and rows repeating by the structured input period
```

The fix is correctness-first: use scalar BF16 gmem/smem copies
(`copy_bits=self.dtype.width`) for both G2S and S2G in the hand-written smoke.

I also aligned two surrounding pieces with local CuTe/quack patterns:

- C staging is now allocated as pointer-swizzled smem (`outer` layout plus
  pointer `swizzle`) instead of passing the composed layout directly.
- The WGMMA accumulator is stored through a quack-style SM90 epilogue retile
  before `stmatrix`.

The scalar G2S/S2G change is the part that flipped the deterministic and random
cases to exact correctness.

## Modal CuTe Stack

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-check
```

App: `ap-OdDXzbiA0H0DzWiOr5dlFL`

Result:

- `cute_viable`: `true`
- `nvidia-cutlass-dsl`: `4.4.2`
- `nvidia-cutlass-dsl-libs-base`: `4.4.2`
- `quack-kernels`: `0.3.10`
- `cuda-python`: `13.2.0`
- `cuda-bindings`: `13.2.0`
- `torch`: `2.13.0.dev20260426+cu132`

## Hand-Written CuTe WGMMA Result

Command:

```bash
GHCR_TAG=785c3fd CPPMEGA_MODAL_GPU=H200 modal run --timestamps \
  scripts/modal_mamba3_mono_chunk_wave2.py --mode cute-gemm
```

App: `ap-SZhp2YPr82ZDy0PzvRwzUM`

Result:

- GPU: `NVIDIA H200`
- Shape: `M=N=K=64`
- dtype: BF16 inputs/output, F32 WGMMA accumulator
- Compile + first launch: `0.82 s`
- `identity_transpose`: `max_abs=0.000000`, `max_rel=0.000000`
- `structured_mod`: `max_abs=0.000000`, `max_rel=0.000000`
- `random_seed_42`: `max_abs=0.000000`, `max_rel=0.000000`
- Correctness: pass
- Timing: `28.254240 us/iter` over `1000` iterations
- Throughput: `0.0186 TFLOP/s`

The random seeded slice now matches the torch reference exactly:

```text
C_out[0,:4] = [6.0625, 10.5625, 11.8750, -11.7500]
C_ref[0,:4] = [6.0625, 10.5625, 11.8750, -11.7500]
```

## Notes

This is still a minimal single-tile smoke. The scalar copies are intentionally
conservative and should not be carried into a performance path unchanged.

Cleanup check:

```bash
modal app list --json
```

All Wave 4 apps were stopped with `0` tasks. The final two relevant apps were:

- `ap-SZhp2YPr82ZDy0PzvRwzUM`: stopped, `0` tasks
- `ap-OdDXzbiA0H0DzWiOr5dlFL`: stopped, `0` tasks

Next CuTe step: keep the scalar-copy version as the correctness oracle, then
reintroduce wider G2S/S2G copies only with a layout-aware copy path for swizzled
smem. Candidate directions are quack's TMA/cp.async mainloop machinery, a
non-swizzled SIMT copy staging path followed by a proven smem layout transform,
or narrower vector widths validated by the deterministic `structured_mod` case.
