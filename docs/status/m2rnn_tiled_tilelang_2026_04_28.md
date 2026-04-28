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
- Inter-tile summary scan is still PyTorch.
- The apply pass is no-local-prefix but sequential inside each tile. This proves the memory path; it is not yet faster than the full PyTorch ParaRNN scan on the small probe.
- TileLang emits a deprecation warning for `TL_DISABLE_TMA_LOWER`; this branch keeps the pass config because it is consistent with the existing TileLang kernels in the repo.
