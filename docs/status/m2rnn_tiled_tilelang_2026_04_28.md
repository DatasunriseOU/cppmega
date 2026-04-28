# M2RNN ParaRNN Tiled TileLang Status - 2026-04-28

## Scope

Worker L worktree:

`/home/dave/source/cppmega/.claude/worktrees/m2rnn-tiled-tilelang`

Branch:

`worker/m2rnn-tiled-tilelang`

Base commit:

`0b7acbc5d18dead10ad206ee5c111e2cb08ab1ef`

## Implemented

- Added `cppmega.megatron.m2rnn_pararnn_tiled_tilelang`.
- Public entry point: `m2rnn_pararnn_tiled_tilelang_forward`.
- Config: `TiledTileLangConfig(max_its, omega_sor, init_strategy, tile_len, backend, allow_tilelang_fallback)`.
- Stats: `TiledTileLangStats` reports backend, TileLang summary/apply attempt/use, max conceptual tile Jacobian elements, actual PyTorch materialized tile Jacobian elements, avoided full Jacobian elements, summary storage, and compile logs.
- Tile lengths are constrained to `16`, `32`, or `64`.
- cuTile is not used.

## Algorithm

Each Newton iteration uses two tiled assembly passes:

1. Summary pass:
   - Assemble one token at a time inside the tile.
   - Reduce the tile to an affine summary:
     `delta_tail = A_tile @ carry + b_tile`.
   - Store only `(A_tile, b_tile)` per `(B,H,K,tile)`.

2. Inter-tile carry scan:
   - PyTorch sequential scan over tile summaries.

3. Apply pass:
   - Re-assemble one token at a time inside the tile.
   - Stream the solve as `delta_t = rhs_t - J_t @ delta_{t-1}` seeded by the scanned tile carry.
   - Write full `delta[B*H*K,S,V]`.
   - Do not store a per-token local prefix map and do not materialize `jac[Be,C,V,V]` when TileLang apply is active.

The full `A[B,S,H,K,V,V]` tensor is not materialized. In the TileLang summary/apply path, `J`, `rhs`, `P`, and `b` exist only as CTA-local fragments; the only global per-tile state is `summary_A`, `summary_b`, scanned carries, and output `delta`.

## TileLang Viability

TileLang is installed and importable:

- `tilelang=0.1.8+cuda.gitf309d814`
- path: `/home/dave/tilelang-build/tilelang/__init__.py`
- CUDA visible with torch `2.13.0.dev20260417+cu132`

The TileLang path covers summary and apply for CUDA fp32 solve buffers with `V=16` and `tile_len in {16,32,64}`. The apply pass is a true no-local-prefix streaming path: it recomputes `J/rhs` per token and carries only `delta_prev` across local token steps.

bf16 callers are covered through the existing fp32 solve-buffer policy: q/k/v/W/xf are promoted to fp32 before the TileLang kernels and outputs are cast back to bf16. The TileLang kernels themselves are not bf16-input kernels yet.

The TileLang summary kernel initially failed layout inference when fragment loops mixed `T.Parallel` and serial reads:

- `h_prev: [vi] and [vj]`
- then an affine inverse check failure from parallel stores

After switching the fragment math and stores to serial V=16 loops, the summary kernel compiles and launches. The apply kernel uses the same conservative serial-fragment style. This is compiler-safe in TileLang `0.1.8+cuda.gitf309d814`, but it limits performance because each CTA performs V=16 fragment math serially and launch count is still two TileLang kernels plus a PyTorch summary scan per Newton iteration.

## Probe Results

```text
python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 2 --repeats 5

tile_len=16
backend_used=tilelang-summary+tilelang-apply
out_max_diff_vs_full_pararnn=1.564622e-07
h_max_diff_vs_full_pararnn=1.192093e-07
torch_materialized_tile_jac_elements=0
tiled_ms=4.106
full_pararnn_ms=1.732

tile_len=32
backend_used=tilelang-summary+tilelang-apply
tilelang_summary_used=True
tilelang_apply_used=True
out_max_diff_vs_full_pararnn=1.564622e-07
h_max_diff_vs_full_pararnn=1.192093e-07
be=8 s=65 v_dim=16 n_tiles=3
max_tile_jac_elements=65536
torch_materialized_tile_jac_elements=0
full_jac_elements_avoided=133120
max_tile_jac_bytes_fp32=262144
full_jac_bytes_fp32=532480
summary_a_elements=6144
summary_b_elements=384
tiled_ms=2.247
full_pararnn_ms=1.723

tile_len=64
backend_used=tilelang-summary+tilelang-apply
out_max_diff_vs_full_pararnn=1.564622e-07
h_max_diff_vs_full_pararnn=1.192093e-07
torch_materialized_tile_jac_elements=0
tiled_ms=3.998
full_pararnn_ms=1.734
```

bf16 caller probe:

```text
python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 33 --H 1 --K 2 --V 16 --max-its 2 --dtype bfloat16
backend_used=tilelang-summary+tilelang-apply
out_max_diff_vs_full_pararnn=0.000000e+00
h_max_diff_vs_full_pararnn=0.000000e+00
torch_materialized_tile_jac_elements=0
```

## Memory Accounting

For `Be = B*H*K`, sequence `S`, value dim `V`, tile length `C`:

- Full ParaRNN Jacobian would be `Be*S*V*V` elements.
- Tiled conceptual peak Jacobian work is `Be*C*V*V` elements.
- Tile summaries are `Be*ceil(S/C)*V*V` for `A_tile` and `Be*ceil(S/C)*V` for `b_tile`.

Example from the tile_len=32 CUDA probe above:

- Full Jacobian: `133120` fp32 elements, `532480` bytes.
- Peak conceptual tile Jacobian: `65536` fp32 elements, `262144` bytes.
- Actual PyTorch materialized tile Jacobian: `0` elements.
- Summary A/B: `6144 + 384` fp32 elements.

For larger `S`, peak conceptual tile work remains bounded by `C`; summary storage scales with number of tiles, not tokens. With TileLang summary+apply active, `torch_materialized_tile_jac_elements=0`; the PyTorch fallback still materializes one tile's `jac` during summary/apply.

## Validation

Passed:

```text
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py
```

Passed:

```text
python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
9 passed
```

Passed:

```text
python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
22 passed
```

Passed:

```text
python -m pytest tests/test_preflight_smem_check.py -q
16 passed
```

## Remaining Limits

- TileLang kernels are fixed to `V=16`, CUDA, fp32 solve buffers, and tile lengths `16/32/64`.
- bf16 caller coverage exists via fp32 staging, not native bf16 TileLang kernel tensors.
- Inter-tile summary scan now has a GPU path: Triton first, TileLang scan fallback,
  then PyTorch fallback if GPU scan compilation/runtime fails.
- The apply pass is no-local-prefix but sequential inside each tile. This proves the memory path; it is not yet faster than the full PyTorch ParaRNN scan on the small probe.
- TileLang emits a deprecation warning for `TL_DISABLE_TMA_LOWER`; this branch keeps the pass config because it is consistent with the existing TileLang kernels in the repo.

## Continuation Update - M2RNN TileLang Scan Optimization

### TileLang/Triton Docs and Local Examples Read

- Web docs: <https://tilelang.com/autoapi/tilelang/language/loop/index.html>,
  <https://www.tilelang.com/autoapi/tilelang/analysis/nested_loop_checker/index.html>,
  <https://www.tilelang.com/autoapi/tilelang/layout/fragment/index.html>,
  <https://tilelang.com/autoapi/tilelang/jit/index.html>,
  <https://www.tilelang.com/get_started/targets.html>,
  <https://www.tilelang.com/programming_guides/autotuning.html>.
- Local docs/examples:
  `/home/dave/tilelang-build/docs/programming_guides/control_flow.md`,
  `/home/dave/tilelang-build/docs/get_started/targets.md`,
  `/home/dave/tilelang-build/docs/get_started/Installation.md`,
  `/home/dave/tilelang-build/examples/gdn/example_cumsum.py`,
  `/home/dave/tilelang-build/tilelang/language/reduce_op.py`,
  `/home/dave/tilelang-build/tilelang/jit/__init__.py`,
  `/home/dave/tilelang-build/tilelang/cache/__init__.py`.
- Takeaways:
  1. `T.serial` is safest inside fragment-heavy recurrence code; nested/adjacent
     `T.Parallel` has semantic restrictions and can trigger layout inference
     failures when mixed with tile ops.
  2. `T.Fragment` controls register/fragment mapping and can be attached to
     parallel loop layout, but the current V=16 recurrence stays conservative
     with serial fragment loops.
  3. TileLang exposes `T.cumsum` for local block prefix work, as shown in
     `examples/gdn/example_cumsum.py`, but affine matrix summary scan is custom.
  4. TileLang JIT/cache is keyed through `tilelang.cache.cached`; target strings
     keep keys deterministic, and compile flags/pass configs flow through
     `tilelang.jit`.
  5. The default compile cache is `~/.tilelang/cache`; `TL_DISABLE_TMA_LOWER`
     is deprecated in this TileLang build but remains used by existing kernels.

### Profiling Findings

Baseline before this patch, command:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 2 --repeats 5
```

Prior status result:

```text
tile_len=16 tiled_ms=4.106
tile_len=32 tiled_ms=2.247
tile_len=64 tiled_ms=3.998
```

Stage timing on GB10 showed why `tile_len=32` is around 2.2 ms: per Newton
iteration, the summary kernel dominates.

```text
S=65, Be=8, V=16
tile_len=16 n_tiles=5 summary_gpu_ms=1.2759 apply_gpu_ms=0.0281 scan_torch_wall_ms=0.1198 scan_triton_wall_ms=0.0055 full_forward_wall_ms=3.9797
tile_len=32 n_tiles=3 summary_gpu_ms=0.7098 apply_gpu_ms=0.0550 scan_torch_wall_ms=0.0746 scan_triton_wall_ms=0.0055 full_forward_wall_ms=2.2209
tile_len=64 n_tiles=2 summary_gpu_ms=1.2128 apply_gpu_ms=0.1090 scan_torch_wall_ms=0.0514 scan_triton_wall_ms=0.0054 full_forward_wall_ms=3.9637
```

So `tile_len=32` is best because it balances fewer tiles than 16 with less
serial per-CTA work than 64. Launch/JIT is not in the hot timing after warmup;
allocation was about 0.004 ms for the small probe. The remaining major cost is
the serial V=16 fragment summary recurrence, especially the `P_next = -J @ P`
work inside every token.

### Patch

- Added a PyTorch-free GPU inter-tile scan:
  Triton first, TileLang scan fallback, PyTorch fallback.
- Kept the TileLang summary/apply kernels and no-local-prefix apply path intact.
- Kept bf16 callers on the existing fp32 solve-buffer policy; outputs still cast
  back to the caller dtype.
- cuTile is not used.

After patch:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 50 --dtype float32

tile_len=16 backend_used=tilelang-summary+triton-scan+tilelang-apply tiled_ms=4.065 full_pararnn_ms=1.713
tile_len=32 backend_used=tilelang-summary+triton-scan+tilelang-apply tiled_ms=2.216 full_pararnn_ms=1.704
tile_len=64 backend_used=tilelang-summary+triton-scan+tilelang-apply tiled_ms=3.977 full_pararnn_ms=1.704
max error vs full ParaRNN: out=1.564622e-07, h=1.192093e-07
```

bf16 caller probe:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 33 --H 1 --K 2 --V 16 --max-its 2 --dtype bfloat16

backend_used=tilelang-summary+triton-scan+tilelang-apply
out_max_diff_vs_full_pararnn=0.000000e+00
h_max_diff_vs_full_pararnn=0.000000e+00
torch_materialized_tile_jac_elements=0
```

### Validation

Passed:

```text
PYTHONPATH=. python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
PYTHONPATH=. python -m pytest tests/test_preflight_smem_check.py -q
```

Observed:

```text
9 passed
22 passed
16 passed
```

### Remaining Limits After This Patch

- Summary/apply remain fixed to CUDA fp32 solve buffers with `V=16` and
  `tile_len in {16,32,64}`.
- Triton scan compiles per `n_tiles` constexpr; TileLang scan and PyTorch scan
  remain fallback paths.
- The path is still slower than full PyTorch ParaRNN on the small probe because
  summary dominates; the next useful optimization is reducing or parallelizing
  the serial V=16 summary fragment work.

## Third Optimization Cycle - Summary Kernel Structure

### TileLang Docs, Examples, and Issues Read

- Official docs:
  <https://www.tilelang.com/get_started/overview.html>,
  <https://tilelang.com/autoapi/tilelang/language/loop/index.html>,
  <https://www.tilelang.com/programming_guides/instructions.html>,
  <https://www.tilelang.com/autoapi/tilelang/layout/fragment/index.html>,
  <https://tilelang.com/tutorials/debug_tools_for_tilelang.html>.
- GitHub/release references:
  <https://github.com/tile-ai/tilelang/releases> (noted `T.cumsum`,
  flexible parallel/local-buffer-in-`T.Parallel`, and layout inference changes),
  <https://github.com/tile-ai/tilelang/releases/tag/v0.1.9>,
  <https://github.com/tile-ai/tilelang/pull/1426>,
  <https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md>.
- Local docs/examples/source:
  `/home/dave/tilelang-build/docs/get_started/overview.md`,
  `/home/dave/tilelang-build/docs/programming_guides/language_basics.md`,
  `/home/dave/tilelang-build/docs/programming_guides/instructions.md`,
  `/home/dave/tilelang-build/docs/deeplearning_operators/gemv.md`,
  `/home/dave/tilelang-build/examples/gdn/example_cumsum.py`,
  `/home/dave/tilelang-build/examples/gdn/example_wy_fast.py`,
  `/home/dave/tilelang-build/tilelang/language/loop.py`,
  `/home/dave/tilelang-build/tilelang/language/reduce_op.py`,
  `/home/dave/tilelang-build/tilelang/layout/fragment.py`,
  `/home/dave/tilelang-build/tilelang/language/copy_op.py`.
- Takeaways:
  1. `T.Parallel(..., loop_layout=T.Fragment(...))` can explicitly annotate
     fragment layout, but nested parallel layout must be attached only to the
     outermost loop and must match input dimensionality. This matches the
     earlier mixed serial/parallel layout-inference blocker, so this cycle did
     not reintroduce parallel fragment writes in the recurrence.
  2. `T.cumsum` is useful for local additive prefix scans, but the tile summary
     is an affine matrix recurrence, so it does not directly replace the custom
     `P,b` recurrence.
  3. `T.copy(..., loop_layout=...)` and shared staging are documented/local
     patterns for reducing global-memory pressure without adding fragment
     layout constraints to the recurrence math.

### Profiling Before Patch

Baseline command:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 30 --dtype float32
```

Before patch:

```text
tile_len=16 tiled_ms=4.020 full_pararnn_ms=1.714
tile_len=32 tiled_ms=2.318 full_pararnn_ms=1.714
tile_len=64 tiled_ms=3.973 full_pararnn_ms=1.709
max error vs full ParaRNN: out=1.564622e-07, h=1.192093e-07
```

Manual stage timing before patch:

```text
tile_len=16 n_tiles=5 summary_gpu_ms=1.3024 scan_triton_gpu_ms=0.0061 apply_gpu_ms=0.0280
tile_len=32 n_tiles=3 summary_gpu_ms=0.7160 scan_triton_gpu_ms=0.0061 apply_gpu_ms=0.0550
tile_len=64 n_tiles=2 summary_gpu_ms=1.2144 scan_triton_gpu_ms=0.0061 apply_gpu_ms=0.1090
```

### Patch

- Rewrote TileLang summary/apply math to use the structure of
  `-J = fI + (1-f)diag(sech2)W^T` directly, avoiding the local `J[V,V]`
  fragment and its serial construction.
- Staged each CTA's `W_be[be,:,:]` into shared memory once with `T.copy`, then
  reused it for `z`, summary `P/b`, and apply `delta` dot products.
- Added `--stage-breakdown`, `--stage-warmup`, and `--stage-repeats` to
  `tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py` for reproducible kernel
  stage timing.
- Tried precomputing `(1-f)*sech2` into an `alpha[V]` fragment; it compiled but
  worsened register/layout pressure (`tile32` summary around 1.54 ms), so that
  sub-attempt was not kept.

### Results After Patch

Stage breakdown command:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --dtype float32 --stage-breakdown --stage-warmup 10 --stage-repeats 50
```

Observed:

```text
stage_breakdown tile_len=32 n_tiles=3 summary_gpu_ms=0.6441 scan_triton_gpu_ms=0.0062 apply_gpu_ms=0.0404
out_max_diff_vs_full_pararnn=1.490116e-07
h_max_diff_vs_full_pararnn=1.192093e-07
```

Manual single-kernel timing after patch was slightly lower when run standalone:

```text
tile_len=32 n_tiles=3 summary_gpu_ms=0.6043 scan_triton_gpu_ms=0.0061 apply_gpu_ms=0.0344
```

End-to-end benchmark after patch:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 30 --dtype float32

tile_len=16 tiled_ms=3.627 full_pararnn_ms=1.687
tile_len=32 tiled_ms=2.046 full_pararnn_ms=1.688
tile_len=64 tiled_ms=3.566 full_pararnn_ms=1.688
max error vs full ParaRNN: out=1.490116e-07, h=1.192093e-07
```

bf16 caller probe still uses fp32 solve buffers:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 33 --H 1 --K 2 --V 16 --max-its 2 --dtype bfloat16

backend_used=tilelang-summary+triton-scan+tilelang-apply
out_max_diff_vs_full_pararnn=0.000000e+00
h_max_diff_vs_full_pararnn=0.000000e+00
torch_materialized_tile_jac_elements=0
```

### Validation

Passed:

```text
PYTHONPATH=. python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
```

Observed:

```text
9 passed
22 passed
```

### Remaining Blockers

- The summary kernel is still serial over the V=16 recurrence inside each CTA.
  Explicit `T.Parallel` fragment writes remain risky without a carefully
  constructed matching `loop_layout=T.Fragment(...)`; the prior mixed
  parallel/serial attempt failed layout inference.
- `T.cumsum` does not directly apply to the affine matrix summary recurrence.
- The path is closer but still slower than full PyTorch ParaRNN for the tiny
  `S=65` probe; summary remains the dominant stage.

## Optimization Cycle 4 - Explicit Layout Probe and V-Reduction Unroll

### Sources Read

- Web/MCP:
  - <https://tilelang.com/autoapi/tilelang/language/loop/index.html>
  - <https://tilelang.com/autoapi/tilelang/analysis/fragment_loop_checker/index.html>
  - <https://github.com/tile-ai/tilelang/pull/1539>
- Local TileLang:
  - `/home/dave/tilelang-build/tilelang/language/loop.py`
  - `/home/dave/tilelang-build/tilelang/layout/fragment.py`
  - `/home/dave/tilelang-build/docs/programming_guides/control_flow.md`
  - `/home/dave/tilelang-build/src/transform/parallel_loop_layout_validator.h`
  - `/home/dave/tilelang-build/src/op/parallel.cc`

Takeaways:

1. `T.Parallel(..., loop_layout=T.Fragment(...))` attaches the
   `parallel_loop_layout` annotation to the outermost parallel loop. The
   fragment `InputDim` must match the number of nested parallel extents.
2. After layout inference, all parallel loops need a layout annotation, and
   nested annotations are only legal on the outermost parallel loop.
3. Fragment loops have stricter rules than shared/global loops. The local
   `fragment_loop_checker` docs call out non-symbolic range requirements, and
   the C++ `ParallelOpNode::ValidateCandidateAgainstFragments` path rejects
   inconsistent fragment access maps.

### Profiling Before Patch

Cycle-local baseline:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 50 --stage-breakdown --stage-warmup 10 --stage-repeats 50 --dtype float32

stage_breakdown tile_len=32 n_tiles=3 summary_gpu_ms=0.6249 scan_triton_gpu_ms=0.0063 apply_gpu_ms=0.0345
tiled_ms=2.355
full_pararnn_ms=1.719
out_max_diff_vs_full_pararnn=1.490116e-07
h_max_diff_vs_full_pararnn=1.192093e-07
```

### Patch

- Added explicit `T.unroll(V)` to the innermost V=16 reduction loops in
  TileLang summary/apply:
  - `z = X + h_prev @ W`
  - `P_next = f*P + (1-f)*diag(sech2)W^T P`
  - `b_next = rhs + f*b + (1-f)*diag(sech2)W^T b`
  - `delta_cur = rhs + f*delta_prev + (1-f)*diag(sech2)W^T delta_prev`
- Added `--sweep-S` to the probe harness so larger sequence-length stage
  timing can be captured in one run.
- Added `tools/probes/tilelang_m2rnn_fragment_parallel_probe.py`, a minimal
  explicit `loop_layout=T.Fragment(...)` probe for the VxV summary update.

The production summary/apply kernels still keep no local `J` and no outside
tile `J`. cuTile is not used.

### Explicit Fragment Layout Blocker

The minimal explicit-layout probe:

```text
PYTHONPATH=. python tools/probes/tilelang_m2rnn_fragment_parallel_probe.py
```

fails during `LayoutInference`, before runtime:

```text
tvm.error.InternalError: Check failed: (StructuralEqual()(it->second.indices, indices)) is false: P: [vk, vj] and [vi, vj]
```

This is with an explicit two-dimensional layout:

```python
T.Fragment(
    [16, 16],
    forward_thread_fn=lambda i, j: (i * 16 + j) % 128,
    forward_index_fn=lambda i, j: (i * 16 + j) // 128,
)
```

Interpretation: the parallel VxV update wants each `(vi, vj)` lane to read
`P[vk, vj]` across `vk`, while the same fragment is also accessed as
`P[vi, vj]`. TileLang records those as inconsistent access maps for one local
fragment inside the same `T.Parallel` region. This keeps the serial-fragment
summary fallback as the safe production path.

### Results After Patch

Small S, direct comparison:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 50 --stage-breakdown --stage-warmup 10 --stage-repeats 50 --dtype float32

stage_breakdown tile_len=32 n_tiles=3 summary_gpu_ms=0.6220 scan_triton_gpu_ms=0.0063 apply_gpu_ms=0.0346
tiled_ms=2.010
full_pararnn_ms=1.700
out_max_diff_vs_full_pararnn=1.490116e-07
h_max_diff_vs_full_pararnn=1.192093e-07
```

The isolated summary stage is effectively unchanged, which means LLVM/NVRTC
was likely already unrolling or optimizing these constant-size reductions.
End-to-end tile32 improved in this run from 2.355 ms to 2.010 ms, but the stage
numbers show the remaining real blocker is still summary recurrence work, not
scan or apply.

Larger S and tile32/64 stage breakdown:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --sweep-tile-lens --sweep-S 65,129,257 --B 1 --H 2 --K 4 --V 16 --max-its 3 --benchmark --warmup 10 --repeats 50 --stage-breakdown --stage-warmup 10 --stage-repeats 50 --dtype float32
```

Observed summary/apply/end-to-end:

```text
S=65  tile16 summary=1.1288 apply=0.0180 tiled=3.508 full=1.689
S=65  tile32 summary=0.6575 apply=0.0347 tiled=2.087 full=1.682
S=65  tile64 summary=1.1542 apply=0.0683 tiled=3.575 full=1.691

S=129 tile16 summary=2.6829 apply=0.0273 tiled=8.230 full=1.895
S=129 tile32 summary=2.2171 apply=0.0350 tiled=6.835 full=1.897
S=129 tile64 summary=1.2443 apply=0.0682 tiled=3.993 full=1.893

S=257 tile16 summary=6.4579 apply=0.0438 tiled=19.614 full=2.092
S=257 tile32 summary=5.2386 apply=0.0532 tiled=16.012 full=2.095
S=257 tile64 summary=4.4087 apply=0.0679 tiled=13.527 full=2.084
```

For larger S, tile64 wins because it launches fewer summary CTAs despite more
serial work per CTA. Scan remains about `0.006 ms`, and apply remains below
`0.07 ms`; summary dominates.

bf16 caller probe:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py --backend tilelang --no-fallback --tile-len 32 --B 1 --S 33 --H 1 --K 2 --V 16 --max-its 2 --dtype bfloat16 --benchmark --warmup 5 --repeats 20

backend_used=tilelang-summary+triton-scan+tilelang-apply
out_max_diff_vs_full_pararnn=0.000000e+00
h_max_diff_vs_full_pararnn=0.000000e+00
tiled_ms=1.240
full_pararnn_ms=1.037
```

### Validation

Passed:

```text
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
```

Observed:

```text
9 passed
22 passed
```

Expected failing diagnostic probe:

```text
PYTHONPATH=. python tools/probes/tilelang_m2rnn_fragment_parallel_probe.py
```

fails with the `P: [vk, vj] and [vi, vj]` `LayoutInference` blocker above.

## Optimization Cycle 5 - Parallel Summary Workaround

### Sources Read

- Web/MCP:
  - <https://tilelang.com/autoapi/tilelang/analysis/fragment_loop_checker/index.html>
  - <https://tilelang.com/autoapi/tilelang/language/loop/index.html>
  - <https://www.tilelang.com/autoapi/tilelang/layout/fragment/index.html>
  - <https://github.com/tile-ai/tilelang/pull/1495>
- Local TileLang:
  - `/home/dave/tilelang-build/docs/get_started/overview.md`
  - `/home/dave/tilelang-build/examples/gdn/example_chunk_o.py`
  - `/home/dave/tilelang-build/examples/kda/chunk_inter_solve_fused.py`
  - `/home/dave/tilelang-build/tilelang/language/loop.py`
  - `/home/dave/tilelang-build/tilelang/layout/fragment.py`

Takeaways:

1. `T.Parallel(..., loop_layout=T.Fragment(...))` must cover the outermost
   parallel loop and the fragment `InputDim` must match the loop nest rank.
2. Fragment buffers inside `T.Parallel` are still subject to stricter access
   validation; simply splitting `P_old` and `P_new` as two register fragments
   does not make a reduction-axis read pattern legal.
3. Moving the old prefix matrix out of fragment memory and into shared memory
   avoids the fragment access-map conflict while keeping the new VxV result in
   a parallel fragment store.

### Minimal Probes

The original mixed-access diagnostic still fails as expected:

```text
PYTHONPATH=. python tools/probes/tilelang_m2rnn_fragment_parallel_probe.py --variant mixed

tvm.error.InternalError:
Check failed: (StructuralEqual()(it->second.indices, indices)) is false:
P: [vk, vj] and [vi, vj]
```

The two-register-fragment coefficient rewrite removes the direct `[vi,vj]`
read but still fails layout inference:

```text
PYTHONPATH=. python tools/probes/tilelang_m2rnn_fragment_parallel_probe.py --variant coeff

tvm.error.InternalError:
Check failed: (analyzer_->CanProveEqual(abs(source->scale), 1)) is false
```

The practical workaround compiles and launches:

```text
PYTHONPATH=. python tools/probes/tilelang_m2rnn_fragment_parallel_probe.py --variant shared-old

fragment_parallel_probe=ok variant=shared-old
```

Interpretation: `P_old` as shared memory plus `P_next` as a fragment is a real
TileLang workaround for this recurrence. The pure two-fragment register
workaround remains blocked in `tilelang=0.1.8+cuda.gitf309d814`.

### Patch

- Added `summary_variant` to `TiledTileLangConfig`:
  - `serial` keeps the previous conservative register-fragment summary path.
  - `parallel_shared_old` is opt-in and uses shared memory for `P_old`/`b_old`,
    computes `P_next[vi,vj]` in `T.Parallel(V,V)`, then copies the result back
    for the next token.
- Added `--summary-variant` to
  `tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py`.
- The probe now reports the Triton recurrence path, when importable, on the
  same shape as full ParaRNN and TileLang.
- Added a CUDA correctness test for the opt-in summary variant.
- cuTile was not used.

### Measurements

Cycle-local serial baseline:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py \
  --backend tilelang --no-fallback --tile-len 32 --summary-variant serial \
  --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --dtype float32 \
  --stage-breakdown --stage-warmup 10 --stage-repeats 30 \
  --benchmark --warmup 5 --repeats 20

stage_breakdown tile_len=32 n_tiles=3 summary_gpu_ms=0.6321 scan_triton_gpu_ms=0.0064 apply_gpu_ms=0.0348
tiled_ms=2.374
full_pararnn_ms=1.713
triton_scan_ms=0.038
```

Opt-in shared-old result, same shape:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py \
  --backend tilelang --no-fallback --tile-len 32 --summary-variant parallel_shared_old \
  --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --dtype float32 \
  --stage-breakdown --stage-warmup 10 --stage-repeats 30 \
  --benchmark --warmup 5 --repeats 20

stage_breakdown tile_len=32 n_tiles=3 summary_gpu_ms=0.0474 scan_triton_gpu_ms=0.0066 apply_gpu_ms=0.0467
tiled_ms=0.366
full_pararnn_ms=1.701
triton_scan_ms=0.036
out_max_diff_vs_full_pararnn=1.490116e-07
h_max_diff_vs_full_pararnn=1.192093e-07
tilelang_out_max_diff_vs_triton=4.917383e-06
tilelang_h_max_diff_vs_triton=5.245209e-06
```

Clean fp32 sweep on `S=65`:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py \
  --backend tilelang --no-fallback --sweep-tile-lens \
  --summary-variant parallel_shared_old \
  --B 1 --S 65 --H 2 --K 4 --V 16 --max-its 3 --dtype float32 \
  --stage-breakdown --stage-warmup 10 --stage-repeats 50 \
  --benchmark --warmup 10 --repeats 50

tile_len=16 summary_gpu_ms=0.0211 apply_gpu_ms=0.0181 tiled_ms=0.155 full_pararnn_ms=1.714 triton_scan_ms=0.066
tile_len=32 summary_gpu_ms=0.0410 apply_gpu_ms=0.0350 tiled_ms=0.582 full_pararnn_ms=1.688 triton_scan_ms=0.066
tile_len=64 summary_gpu_ms=0.1281 apply_gpu_ms=0.1158 tiled_ms=1.028 full_pararnn_ms=1.677 triton_scan_ms=0.036
```

bf16 caller probe:

```text
PYTHONPATH=. python tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py \
  --backend tilelang --no-fallback --tile-len 32 \
  --summary-variant parallel_shared_old \
  --B 1 --S 33 --H 1 --K 2 --V 16 --max-its 2 --dtype bfloat16 \
  --benchmark --warmup 5 --repeats 20

backend_used=tilelang-summary-parallel-shared-old+triton-scan+tilelang-apply
out_max_diff_vs_full_pararnn=0.000000e+00
h_max_diff_vs_full_pararnn=0.000000e+00
tiled_ms=0.374
full_pararnn_ms=1.053
triton_scan_ms=0.045
```

### Validation

Passed:

```text
PYTHONPATH=. python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py tools/probes/tilelang_m2rnn_fragment_parallel_probe.py
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
PYTHONPATH=. python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
```

Observed:

```text
10 passed
23 passed
```

### Recommendation

Do not mark the TileLang branch backup-only anymore. The shared-old workaround
turns the summary stage from the bottleneck into a small cost and makes the
tiled TileLang path faster than full ParaRNN on the probe. Keep it opt-in for
now because it uses more shared memory and the exact Triton recurrence path is
still faster on these small shapes. Next useful work is to promote
`parallel_shared_old` to the default only after a larger NAM56R-shaped memory
and occupancy sweep.

## Optimization Cycle 6 - Shared-Old Default Candidate

### Sources Read

- Web docs:
  - <https://www.tilelang.com/programming_guides/instructions.html>
  - <https://tilelang.com/autoapi/tilelang/language/copy/index.html>
  - <https://tilelang.com/autoapi/tilelang/language/loop/index.html>
  - <https://www.tilelang.com/autoapi/tilelang/layout/fragment/index.html>
  - <https://www.tilelang.com/programming_guides/autotuning.html>
- Local TileLang:
  - `/home/dave/tilelang-build/examples/linear_attention/example_mamba_chunk_scan.py`
  - `/home/dave/tilelang-build/examples/kda/chunk_inter_solve_fused.py`
  - `/home/dave/tilelang-build/examples/gemm/example_gemm_autotune.py`
  - `/home/dave/tilelang-build/tilelang/language/loop.py`
  - `/home/dave/tilelang-build/tilelang/language/copy_op.py`
  - `/home/dave/tilelang-build/tilelang/transform/pass_config.py`

Takeaways:

1. `T.copy` is the right primitive for global/shared/fragment tile movement;
   TileLang handles coalescing and safe boundary legalization, with `loop_layout`
   available for SIMT copies.
2. `T.Parallel(..., loop_layout=T.Fragment(...))` is supported, but the fragment
   input dimensionality and access maps must be consistent. This matches the
   previous layout-inference failures for mixed `P[vk,vj]` and `P[vi,vj]`.
3. TileLang autotune exists, but this kernel's useful choice space is currently
   small (`tile_len` 16/32/64 and summary variant). A static default is lower
   risk until apply-side work is improved.

### Profiling

Environment:

```text
device=NVIDIA GB10
torch=2.13.0.dev20260417+cu132
tilelang=0.1.8+cuda.gitf309d814
triton=importable
```

Serial summary baseline:

```text
S=512 Be=16 fp32
tile16 summary=68.0183ms apply=0.3084ms tiled=206.829ms full=25.617ms triton=0.700ms
tile32 summary=52.1386ms apply=0.1775ms tiled=199.107ms full=24.644ms triton=0.705ms
tile64 summary=63.4690ms apply=0.3249ms tiled=193.384ms full=25.434ms triton=0.715ms
```

Shared-old summary:

```text
S=128 Be=16 fp32
tile16 summary=0.0527ms apply=0.0449ms tiled=0.797ms full=3.533ms triton=0.527ms
tile32 summary=0.0653ms apply=0.0536ms tiled=0.867ms full=3.261ms triton=0.525ms
tile64 summary=0.0818ms apply=0.0702ms tiled=1.451ms full=3.233ms triton=0.524ms

S=512 Be=16 fp32
tile16 summary=0.3046ms apply=0.2695ms tiled=2.500ms full=20.526ms triton=0.649ms
tile32 summary=0.3129ms apply=0.2877ms tiled=2.594ms full=19.825ms triton=0.649ms
tile64 summary=0.4569ms apply=0.3001ms tiled=2.679ms full=19.856ms triton=0.660ms
```

Larger bf16 caller sweep:

```text
S=512 Be=64 bf16
tile16 summary=1.6419ms apply=1.2184ms tiled=8.988ms full=119.399ms triton=0.660ms
tile32 summary=1.6089ms apply=1.2109ms tiled=9.598ms full=119.422ms triton=0.660ms
tile64 summary=1.6390ms apply=1.3741ms tiled=9.769ms full=119.870ms triton=0.640ms

S=1024 Be=64 bf16
tile16 summary=4.1703ms apply=3.2549ms tiled=22.874ms full=315.767ms triton=1.589ms
tile32 summary=4.1475ms apply=3.2557ms tiled=22.879ms full=317.873ms triton=1.584ms
tile64 summary=3.7641ms apply=3.1130ms tiled=23.719ms full=315.724ms triton=1.488ms
```

After warm cache, default probe now selects shared-old without an explicit
`--summary-variant`:

```text
backend_used=tilelang-summary-parallel-shared-old+triton-scan+tilelang-apply
summary_gpu_ms=0.0418 scan_triton_gpu_ms=0.0017 apply_gpu_ms=0.0358
tiled_ms=0.719 full_pararnn_ms=2.170 triton_scan_ms=0.509
```

### Patch

- Made `TiledTileLangConfig.summary_variant` default to
  `parallel_shared_old`.
- Updated the probe CLI default to match the production config.
- Added `_try_tilelang_summary_with_serial_fallback`: if shared-old compile or
  launch fails, the old serial TileLang summary kernel is tried before falling
  back to PyTorch.
- Updated CUDA backend assertions to allow the new default backend string.
- Added a CPU unit test for shared-old -> serial TileLang summary fallback.
- Tried a parallel/shared apply variant, but it produced TileLang ThreadSync
  warnings and no measured speedup, so it was not kept.

### Correctness and Validation

Observed correctness:

```text
fp32 S=512 Be=16 max diff vs full ParaRNN:
out <= 2.682209e-07, h <= 9.313226e-08

bf16 S=1024 Be=64 max diff vs full ParaRNN:
out <= 1.953125e-03, h <= 1.907349e-06
```

Passed:

```text
python -m py_compile cppmega/megatron/m2rnn_pararnn_tiled_tilelang.py tools/probes/m2rnn_pararnn_tiled_tilelang_probe.py
python -m pytest tests/test_m2rnn_pararnn_tiled_tilelang.py -q
python -m pytest tests/test_m2rnn_pararnn.py tests/test_m2rnn_pararnn_tiled_tilelang.py -q
```

Observed:

```text
11 passed
24 passed
```

### Default Recommendation

`parallel_shared_old` should be the default for the current eligible TileLang
path (`CUDA`, fp32 solve buffers, `V=16`, `tile_len in {16,32,64}`). It is
orders of magnitude faster than the serial TileLang summary and is stable
across fp32/bf16 caller probes. Keep the serial TileLang summary and PyTorch
paths as fallbacks.

It is not yet a full Triton replacement. Triton recurrence is still materially
faster in the tested shapes, especially larger `Be/S`. The next production
blocker is apply-side and per-token recurrence work, not the inter-tile scan.
