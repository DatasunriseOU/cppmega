# Mamba3 Hopper True WS Probe - 2026-04-29

Branch: `worker/mamba3-hopper-true-ws`

Base: `worker/mamba3-hopper-tma-ws-fix` (`ccd9679`)

Goal: explain why TileLang logged `[WS] skipped: no TMA copies in pipeline loop`
for the previous `qk_shared_direct` compile+smoke path, then find the smallest
non-production changes that make producer/consumer warp specialization really
fire for Mamba3 MIMO `bwd_fwd` and `bwd_bwd`.

## Sources Checked

Local TileLang source:

- `/home/dave/tilelang-build/src/transform/producer_consumer_ws.cc`
- `/home/dave/tilelang-build/tilelang/engine/phase.py`
- `/home/dave/tilelang-build/testing/python/issue/test_tilelang_issue_tma_no_ws.py`
- `/home/dave/tilelang-build/testing/python/language/test_tilelang_language_tma_copy.py`

External MCP web search:

- Exa found TileLang commit `ded6a99` / PR #1909: producer-consumer WS and
  `T.tma_copy()` API.
  <https://github.com/tile-ai/tilelang/commit/ded6a992218a5ec0dbb490128cb21aae7794cdb8>
- Tavily found TileLang 0.1.9 instruction docs and the TileLang paper; both are
  consistent with Hopper producer warps owning async TMA loads while consumer
  warps own compute.
  <https://tilelang.com/programming_guides/instructions.html>
  <https://arxiv.org/pdf/2504.17577>
- Perplexity summary matched local source: auto WS requires TMA producer copies
  in a `T.Pipelined(..., num_stages >= 1)` loop.

## Why WS Was Skipped

The relevant pass order in `tilelang/engine/phase.py` is:

1. `ProducerConsumerWarpSpecialized()`
2. `PipelinePlanning()`
3. `InjectSoftwarePipeline()`
4. `LayoutInference()`
5. `LowerTileOp()`

So the WS pass runs early on high-level tile-op IR. It does not wait for later
pipeline planning to create TMA copies.

In `producer_consumer_ws.cc`, `TiledWSCandidate::Check` requires both:

- a `For` loop annotated with `num_stages >= 1`;
- a `T.copy`/`T.tma_copy` inside that loop that `ClassifyCopy` can prove is a
  TMA load producer.

The previous Modal probe called:

```python
mod.mamba_mimo_bwd_fwd(..., "bfloat16")
mod.mamba_mimo_bwd_bwd(..., "bfloat16", 256, 0)
```

`mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd` default to `num_stages=0`. With
`num_stages=0`, the candidate scanner never enters `in_pipeline_`, so it emits:

```text
[WS] skipped: no TMA copies in pipeline loop
```

This was not a TMA-shape failure in `qk_shared_direct`; it was a missing pipeline
annotation at the call site. The existing `qk_shared_direct` source workaround is
still needed because the original 3D/`qk_dot_frag` path hits the earlier
FloorMod/layout blockers before it becomes useful.

## New Artifacts

- `scripts/modal_mamba3_true_ws_probe.py`
- `upstream_prs/examples/13_tilelang_floormod_dbz/mamba3_bwd_true_ws_stage1_nonproduction.patch`

The patch is explicitly non-production. It combines the previous rank-2
TMA/qk-shared-direct workaround with `bf_num_stages=1` and `bb_num_stages=1`
defaults in the Python wrapper. The Modal probe does not patch production files;
it applies equivalent edits to a temp copy.

Patch validation:

```text
patch --dry-run -p4 /tmp/.../mamba3_mimo_bwd.py < mamba3_bwd_true_ws_stage1_nonproduction.patch
checking file /tmp/.../mamba3_mimo_bwd.py
```

## Modal Compile Results

Modal app runs:

- H100: <https://modal.com/apps/jewelmusic/main/ap-neKwIAECjAlI19B3INjkm9>
- H200: <https://modal.com/apps/jewelmusic/main/ap-F8sjkUEfVTdMYhmjTje5z2>

Common environment:

- image: `ghcr.io/jewelmusicee/cppmega:latest`
- TileLang: `0.1.8+cu132.gitf309d814`
- Torch: `2.13.0.dev20260426+cu132`
- CUDA: `13.2`

### H100 Compile

Actual device: `NVIDIA H100 80GB HBM3`, capability `(9, 0)`.

| `bf_num_stages` | `bb_num_stages` | bwd_fwd WS | bwd_bwd WS | Notes |
| --- | --- | --- | --- | --- |
| 0 | 0 | no | no | Both compile; pass logs `skipped`. |
| 1 | 0 | yes | no | `bwd_fwd`: `candidate found`, producer guard `if (128 <= threadIdx.x)`, `tl::tma_load` x5. |
| 0 | 1 | no | yes | `bwd_bwd`: `candidate found`, producer guard `if (256 <= threadIdx.x)`, `tl::tma_load` x8. |
| 1 | 1 | yes | yes | Both WS. |
| 2 | 2 | yes | yes | Compile OK; runtime fails in smoke. |

### H200 Compile

Actual device: `NVIDIA H200`, capability `(9, 0)`.

| `bf_num_stages` | `bb_num_stages` | bwd_fwd WS | bwd_bwd WS | Notes |
| --- | --- | --- | --- | --- |
| 0 | 0 | no | no | Launch bounds stay `128` / `256`. |
| 1 | 0 | yes | no | `bwd_fwd` launch bound becomes `256`, producer guard present. |
| 0 | 1 | no | yes | `bwd_bwd` launch bound becomes `512`, producer guard present. |
| 1 | 1 | yes | yes | Both WS. |
| 2 | 2 | yes | yes | Compile OK; runtime fails in smoke. |

Representative pass log for `num_stages=1`:

```text
[WS] candidate found, applying MVB + WS
[WS] transformation applied successfully
```

## Modal Smoke Results

Smoke shape:

```text
B=1, S=64, H=4, G=1, N=64, P=64, R=4, chunk_size=16, dtype=bf16
```

Stage-1 smoke passed on both H100 and H200:

| GPU | `bf_num_stages` | `bb_num_stages` | Result | qk_dot_absmax | dq_absmax | dk_absmax | dv_absmax |
| --- | --- | --- | --- | --- | --- | --- | --- |
| H100 | 1 | 0 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| H100 | 0 | 1 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| H100 | 1 | 1 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| H200 | 1 | 0 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| H200 | 0 | 1 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |
| H200 | 1 | 1 | OK | 0.005767822265625 | 6.148e-10 | 1.295e-09 | 3.449e-09 |

Stage-2 smoke failed on both H100 and H200:

```text
Failed to initialize the TMA descriptor 716
(CUDA_ERROR_MISALIGNED_ADDRESS: misaligned address)
format CU_TENSOR_MAP_DATA_TYPE_FLOAT32
dim 3
globalDim [64, 4, 1]
globalStridesRaw [4, 256, 1024]
boxDim [16, 1, 1]
```

This descriptor matches the family of float32 `[B, H, S]` per-chunk vector
loads such as `DA_CS` / `DA_CS_REV` / `DT`-like slices. It is separate from the
original Q/K/QK_DOT rank-2 workaround.

## Conclusion

True producer/consumer WS is achievable for both Mamba3 MIMO backward kernels
without a large rewrite, but only after two conditions are met:

1. keep the previous non-production qk/shared-direct rank-2 TMA workaround;
2. call the kernels with `bf_num_stages=1` and/or `bb_num_stages=1`.

`num_stages=1` is the current minimal working setting. `num_stages=2` compiles
and WS fires, but runtime TMA descriptor setup fails on a rank-3 float32 vector
load path.

## Next Steps

1. Benchmark `qk_shared_direct + bf_num_stages=1 + bb_num_stages=1` against the
   `num_stages=0` TMA compile/smoke baseline on H100/H200.
2. Add correctness comparison against the non-TMA baseline over representative
   Mamba3 shapes, not just the small smoke tensors.
3. For stage depth greater than 1, either flatten/reshape the `[B,H,S]` float32
   vector inputs into TMA-legal 2D descriptors or force those small vector
   copies off the TMA path while keeping Q/K/QK_DOT TMA producers.
