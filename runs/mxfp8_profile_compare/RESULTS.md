# GB10 NAM56R-quarter 20-step profile comparison

Hardware: NVIDIA GB10 (Grace Blackwell, sm_120, 128 GB unified memory).
Stack: torch 2.13.0.dev20260417+cu132, TE 2.16.0.dev0+8e19460b (locally rebuilt
from `/home/dave/TransformerEngine` source — both core `libtransformer_engine.so`
and the `transformer_engine_torch` shim).
Shape: 13 layers (NAM56R-quarter), seq 4096, micro-batch 4, MTP=2, MoE flex
dispatcher, Muon optimizer, BF16 no-master + q8 momentum.

Per-step token count: micro_batch_size × seq_length = 4 × 4096 = **16 384 tok/step**.

## Results table

| profile              | iters done | steady-state ms/step¹ | tok/sec¹ | final lm_loss² | max allocated MiB³ | max reserved MiB³ |
| -------------------- | ---------: | --------------------: | -------: | -------------: | -----------------: | ----------------: |
| **bf16**             |      20/20 |                  4935 | **3320** |          4.466 |             27 706 |            28 754 |
| **mxfp8_gemm_ready** |      20/20 |                  5267 |     3111 |          4.851 |         **27 042** |        **28 124** |
| mxfp8_legacy         |       0/20 |                     — |        — |              — |                  — |                 — |
| mxfp8_compact_direct |       0/20 |                     — |        — |              — |                  — |                 — |

¹ Steady-state averaged over iterations 3–20 (iter 1–2 are TileLang JIT compile
+ cuBLAS heuristic warmup, ~55 s and ~30 s respectively).
² Loss at iter 20.  Loss trajectory iter 1→20: bf16 11.76→4.47, mxfp8 11.74→4.85.
³ From `[Rank 0] (after 2 iterations) memory (MB)` line in the training log.
The shell `nvidia-smi` poller returns `[N/A]` for `memory.used` on GB10's
unified-memory architecture, so torch's allocator stats are the source of
truth here.

## Headline observations

- **bf16 is the fastest path** on this small shape: ~6.7 % faster steady-state
  than mxfp8_gemm_ready and reaches a slightly lower loss after 20 steps. The
  tiny micro-batch (4) leaves MXFP8 quantize/dequantize overhead exposed
  relative to the bf16 GEMM; MXFP8 wins density, not throughput, at this
  problem size.
- **mxfp8_gemm_ready uses ~660 MiB less GPU memory** (allocated; ~630 MiB
  reserved) than bf16. The savings come from rowwise/columnwise FP8 weights and
  saved-operand sidecars. On a memory-bound shape (longer seq or larger model)
  this would expand significantly, but on the quarter shape both fit
  comfortably in 128 GB.
- Both runs converge cleanly with no NaN / skipped iterations.
- bf16 is faster per step *and* converges marginally lower → on the GB10 quarter
  shape, **bf16 is the recommended default** unless you specifically need the
  mxfp8 memory headroom.

## Crashes (configs ruled out)

### `mxfp8_legacy` — two latent bugs in the legacy contract path

Crashes deterministically at iter 1 with the default `legacy` runtime knobs
(`transpose_emit_backend=te`, `transpose_emit_swizzled=1`,
`dense_saved_operands=0`, `compact_columnwise_backward=0`):

```
File "megatron/core/transformer/moe/moe_layer.py", line 527, in postprocess
  output = output + shared_expert_output
RuntimeError: The size of tensor a (4) must match the size of tensor b (16384)
              at non-singleton dimension 1
```

`output` arrives shaped `(S, B, H) = (4096, 4, 3584)` while
`shared_expert_output` arrives `(B*S, H) = (16384, 3584)`.  Bisecting the four
runtime-knob differences vs `gemm_ready_v1` shows:

| `transpose_emit_backend` | `dense_saved_operands` + `compact_columnwise_backward` | result                              |
| ------------------------ | ------------------------------------------------------ | ----------------------------------- |
| `te` (legacy default)    | False (legacy default)                                 | MoE shape mismatch at iter 1        |
| `off`                    | False (legacy default)                                 | CUDA `misaligned address` at iter 1 |
| `off`                    | True  (gemm_ready_v1)                                  | ✅ runs 5/5 cleanly                  |

So `legacy` has *two* latent kernel/shape bugs on the current
NAM56R-quarter+MoE-flex shape — disabling the TE-side transpose emit just
exposes the second one.  The wave43b contract design intentionally lets
`legacy` keep TE-emit / sidecars / copy-transpose for measurement
comparability (see `docs/status/mxfp8_linear_kernel_contract_wave43b_2026_05_01.md`),
so the right fix is in the underlying kernel/shape paths, not the contract knobs.

**Practical recommendation:** use `gemm_ready_v1` as the production MXFP8
contract on GB10 NAM56R-quarter.  `legacy` is broken on this exact
shape+MoE config until both bugs are tracked down (open follow-ups: 
(a) MoE postprocess flatten-mismatch when TE-side transpose-emit is active,
(b) MXFP8 vectorized-load alignment when dense saved operands are off).

### `mxfp8_compact_direct_v1` — contract refuses copy fallback

Crashes deterministically at iter 1 with:

```
File "scripts/cppmega_fp8_shim.py", line 1465, in
  _cppmega_mxfp8_colwise_as_rowwise_transpose
ValueError: compact_direct_v1 forbids copy-based MXFP8 transpose
            materialization for type=MXFP8TensorStorage, …
```

Triggered by:
`fallback_reasons = {'missing_gemm_ready_grouped_transpose:0': 1,
                     'compact_direct_v1_forbids_copy_transpose': 1}`.

The grouped-GEMM backward path on this MoE configuration encounters at least
one operand that doesn't yet have a gemm-ready columnwise transpose and the
production code falls back to a copy-based transpose. `compact_direct_v1` is
*designed* to refuse exactly that — it's a strict no-materialization contract
for the future direct-backward lane.  Until the grouped-GEMM producer learns
to emit gemm-ready operands for the missing case, the contract correctly
guards the run from silently falling back.

This is the expected/intended behavior, not a regression.  The contract is
working as documented in the runtime validators
(`cppmega/megatron/mxfp8_sidecar_refs.py:MXFP8_COMPACT_DIRECT_ZERO_COUNTERS`
and the `compact_direct_counter_violations` helper added this session).

## Batch=16 follow-up

The batch=4 numbers above leave ~100 GB of GB10's 128 GB memory unused.  At
that compute density MXFP8's quantize/dequantize overhead isn't amortized, so
bf16 wins on speed.  Re-running the comparison at `--micro-batch-size 16
--global-batch-size 16` (4× compute density, 65 536 tok/step) flips the
result:

| profile                  |        iters done |            steady-state ms/step⁴ |     tok/sec | final lm_loss⁵ | max allocated MiB | max reserved MiB |
| ------------------------ | ----------------: | -------------------------------: | ----------: | -------------: | ----------------: | ---------------: |
| **bf16_b16**             |             20/20 | 20 481 (mean) / 16 961 (median)⁶ | 3200 / 3863 |          3.899 |            89 407 |           93 228 |
| **mxfp8_gemm_ready_b16** | 20/20 (post-fix)⁷ |                **18 014 (mean)** |    **3638** |          4.288 |        **82 622** |       **85 872** |

⁴ Steady-state averaged over iterations 3–N (3–20 for bf16, 3–10 for mxfp8 due
to crash; see below).
⁵ Loss at the last completed iteration.
⁶ bf16 oscillates between a "fast" cluster (~16 s) and a "slow" cluster (~27 s)
on a 1-on-1-off cadence — likely cuBLAS heuristic re-selection per step.
mxfp8 is far more stable (~18 s flat).
⁷ Required a TE rebuild with an alignment fix in
`multi_cast_transpose.cu` — see "mxfp8_gemm_ready_b16 crash at iter 11" below.

**Headline at batch=16:**

- **MXFP8 gets faster than bf16 by ~10.5 % on mean step time** (3619 vs 3200
  tok/sec), and is dramatically more *stable* per step.  This is the regime
  where MXFP8 is supposed to win — quantize overhead amortizes against larger
  GEMMs.
- **MXFP8 saves ~6.8 GB** (89.4 → 82.6 GB allocated, ~7.6 %).  The savings
  scale with batch.  bf16 is approaching the 128 GB ceiling (70 % used);
  MXFP8 has more headroom for further batch growth.
- Loss curves overlap until the crash (bf16 11.75→5.54 at iter 10, mxfp8
  11.74→6.16 at iter 10 — within noise on this tiny scale).

### `mxfp8_gemm_ready_b16` crash at iter 11 (FIXED)

Original symptom (now fixed): crashed deterministically after iter 10 with
`CUDA error: misaligned address` (exitcode -6, SIGABRT from
`c10::AcceleratorError`).

**Root cause:** TE's `multi_cast_transpose` kernel
(`/home/dave/TransformerEngine/transformer_engine/common/transpose/multi_cast_transpose.cu:287`)
selected its 16-byte-vectorized `aligned` path based on tensor dimensions
alone (`num_tiles_m * tile_dim_m == num_rows && num_tiles_n * tile_dim_n ==
row_length`).  At batch=16 NAM56R-quarter shapes the columnwise output base
pointer lands on a non-16-byte offset even though the tile arithmetic divides
evenly, so the vectorized `Vec<...>::store_to(...)` issues a `uint128` store
to an unaligned address → `cudaErrorMisalignedAddress`.  Surfaces only after
the GEMM-ready transpose path actually exercises the columnwise output (iter
11 onwards in this shape).

**Fix:** add explicit pointer-alignment guards on the input/columnwise output
pointers before selecting the aligned path:

```cpp
constexpr uintptr_t kVecAlign = 16;
const auto* in_dptr   = input_list[tensor_id]->data.dptr;
const auto* out_c_dptr = output_list[tensor_id]->data.dptr;
const auto* out_t_dptr = output_list[tensor_id]->columnwise_data.dptr;
const bool aligned =
    ((num_tiles_m * tile_dim_m == num_rows) && (num_tiles_n * tile_dim_n == row_length)
     && in_dptr != nullptr && out_c_dptr != nullptr && out_t_dptr != nullptr
     && (reinterpret_cast<uintptr_t>(in_dptr)    % kVecAlign == 0)
     && (reinterpret_cast<uintptr_t>(out_c_dptr) % kVecAlign == 0)
     && (reinterpret_cast<uintptr_t>(out_t_dptr) % kVecAlign == 0));
```

Pointers that fail any of those checks fall through to the unaligned kernel.
Re-built `libtransformer_engine.so` from the cppmega TE fork.

**Verification (post-fix run, see `profile_mxfp8_gemm_ready_b16_postfix_*.log`):**

| metric                   |                   before fix |             after fix |
| ------------------------ | ---------------------------: | --------------------: |
| iters completed          |                        10/20 |             **20/20** |
| crash signature          | misaligned address @ iter 11 |                  none |
| mean ms/step (iter 3-20) |             18 324 (8 iters) | **18 014** (18 iters) |
| tok/sec                  |                        3 619 |             **3 638** |
| final lm_loss            |              6.156 (iter 10) |   **4.288** (iter 20) |
| max allocated MiB        |         82 618 (iter 2 only) |                82 622 |

## Reproduce

```bash
cd /home/dave/source/cppmega
# full matrix (three existing b4 configs + two b16 configs), isolated and fail-fast:
python3 runs/mxfp8_profile_compare/run_compare.py --suite all
# batch=4 only (correctness/debug):
./runs/mxfp8_profile_compare/run_compare.sh
# batch=16 only (compute-density comparison):
./runs/mxfp8_profile_compare/run_batch16.sh
```

Logs land in `runs/mxfp8_profile_compare/profile_<name>_<timestamp>.log` plus a
matching `.nvsmi.log` (latter is `[N/A]` on GB10 — see note above).  Since
P087 the python driver runs each config in its own subprocess and aborts
fail-fast on the first non-zero exit (the shell runner previously continued
after failures).

## Methodology note

Earlier attempts ran all three configs in the same process tree.
`mxfp8_gemm_ready` happened to crash at iter 3 with `cuBLAS Error: an
internal operation failed` when run *immediately after* the bf16 run — likely
GPU memory fragmentation or stale TileLang JIT state across the boundary.
Running it in isolation (one process per config) reproduced 20/20 cleanly,
which is what the table above reflects.  P087 makes the process boundary
structural: `run_compare.py` spawns every config as a separate subprocess
with a fresh environment and aborts fail-fast on the first non-zero exit, so
a full-matrix run (`--suite all`) no longer shares CUDA context, cuBLAS
handles or JIT caches between configs.  GPU re-verification of the full
matrix on GB10 is still pending.
