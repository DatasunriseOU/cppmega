# Mamba3 Mono WGMMA Copy Path Wave6 - 2026-04-30

Status: active
Canonical: no
Date: 2026-04-30
Scope: Lane C receipt/model/check for the narrow 128-bit vector copy attempt
and the minimal TMA/cp.async descriptor prototype gates.

Branch: `worker/mamba3-mono-triton-model`

## Receipt

Wave5 established the copy ledger for the monolithic WGMMA/CuTe path:

| item | value |
| --- | ---: |
| large tile movements per chunk | `12` |
| TMA/cp.async-eligible global tiles | `10` |
| CTA-local `dstates` stages | `2` |
| large-copy bytes per chunk | `98304 B` / `96 KiB` |
| global TMA/cp.async-eligible bytes per chunk | `81920 B` / `80 KiB` |
| local stage bytes per chunk | `16384 B` / `16 KiB` |

Wave6 adds the concrete static receipt:

- `tools/probes/mamba3_wgmma_wave6_copy_path.py`
- `tests/test_mamba3_wgmma_wave6_copy_path.py`
- `docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json`

Check command:

```text
python tools/probes/mamba3_wgmma_wave6_copy_path.py \
  --check docs/status/mamba3_mono_wgmma_copy_path_wave6_receipt_2026_04_30.json
```

## Narrow 128-bit Attempt

Static model result: pass, but timing is blocked until compiled ptxas metadata
is attached.

| gate | value |
| --- | ---: |
| tile shape | `64x64 bf16` |
| row bytes | `128` |
| tile bytes | `8192` |
| vector width | `16 B` |
| vectors per row | `8` |
| vectors per tile | `512` |
| row tail | `0 B` |
| tile tail | `0 B` |
| copy instructions per chunk | `6144` |

The proof is conditional on runtime base-pointer guards:

- every vectorized global source pointer has `ptr % 16 == 0`;
- every vectorized SMEM destination pointer has `ptr % 16 == 0`;
- emitted row and panel offsets preserve 16-byte alignment;
- generated code has no masked vector-tail path.

Swizzled-SMEM compatibility is now explicit. The allowed physical SMEM layout
must provide 128-byte rows, and each 16-byte vector lane must remain contiguous
inside that 128-byte swizzle atom. `K_T` and `Q_T` fail the gate if the
transpose operand is implemented as per-column 2-byte SMEM scatter instead of
a vector-compatible physical layout or view.

Required ptxas fields:

| field | pass value |
| --- | ---: |
| `registers_per_thread` | `<=192` |
| `static_smem_bytes + dynamic_smem_bytes` | `<=118784` |
| `spill_stores_bytes` | `0` |
| `spill_loads_bytes` | `0` |

Missing ptxas metadata is a fail for timing entry, even though the static
alignment/tail model passes.

## TMA Descriptor Prototype

Static descriptor model result: plan passes, but implementation smoke and
ptxas metadata are still required before claiming green or yellow timing.

| gate | value |
| --- | ---: |
| descriptor prototypes | `10` |
| descriptor tile shape | `64x64 bf16` |
| descriptor bytes per tile | `8192` |
| expected mbarrier bytes per chunk | `81920` |
| non-TMA large tiles | `dstates_panel_p0`, `dstates_panel_p1` |
| dynamic SMEM target | `131072 B` |

The descriptor plan covers only `K`, `Q`, `K_T`, `Q_T`, `state_panel_p0/1`,
`dPhiO_panel_p0/1`, and `PsiV_panel_p0/1`. CTA-local `dstates` stages and tiny
scalar/vector slices remain non-TMA.

The TMA path must prove descriptor/tensor-map construction, expected-byte
mbarrier accounting, and wait/fence placement before WGMMA consumes SMEM.
The ptxas fields are the same schema as the vector path, but the total SMEM
budget is `<=131072 B` exactly.

## Source Notes

Official docs used for these gates:

- CUDA Programming Guide async copies:
  <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html>
- libcu++ PTX `cp.async.bulk` wrapper:
  <https://nvidia.github.io/cccl/libcudacxx/ptx/instructions/cp_async_bulk.html>
- CUTLASS CuTe TMA tensors:
  <https://docs.nvidia.com/cutlass/4.3.2/media/docs/cpp/cute/0z_tma_tensors.html>
- CuTe DSL cpasync API:
  <https://docs.nvidia.com/cutlass/4.2.1/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_cpasync.html>

The key modeled constraints are: 16-byte source/destination alignment,
16-byte transfer-size multiples, tensor-map/descriptor construction for
multidimensional TMA, shared-memory-barrier completion for global-to-shared TMA
reads, explicit expected-byte accounting for low-level `cp.async.bulk`, and
128B swizzle compatibility for the SMEM tile layout.

## Updated Gates

Pass into timing:

- `narrow_vector_128b_safe_attempt`: static proof passes for all 12 large
  tiles, runtime `ptr % 16` guards exist, no tails are emitted, `K_T/Q_T` use
  vector-compatible SMEM layout/view, ptxas reports `<=192` regs/thread,
  total SMEM `<=118784 B`, and zero spills.
- `tma_cp_async_target`: exactly 10 global tile descriptors, exactly 2 local
  non-TMA `dstates` stages, expected mbarrier bytes `81920 B/chunk`, waits and
  fences before WGMMA use, ptxas reports `<=192` regs/thread, total SMEM
  `<=131072 B`, and zero spills.

Kill:

- any missing 16-byte alignment proof or masked vector tail;
- any transpose operand implemented as scalar 2-byte SMEM scatter on the
  narrow-vector path;
- any TMA descriptor for CTA-local `dstates` or tiny scalar/vector slices;
- any missing mbarrier expected-byte update, wait, or fence on the TMA path;
- ptxas metadata missing for a timing claim, regs/thread `>192`, SMEM over the
  variant budget, or any spill byte.

## Wave7 Tasks

1. Add the actual narrow-vector copy iterator in the monolithic WGMMA skeleton
   and emit runtime alignment assertions for every vectorized source and SMEM
   destination.
2. Compile with `--ptxas-options=-v` and attach the five required ptxas fields
   to the Wave6 receipt: registers/thread, static SMEM, dynamic SMEM,
   spill stores, and spill loads.
3. Inspect generated code for `K_T/Q_T`: prove the transpose operands use a
   vector-compatible SMEM layout/view, not per-column scalar scatter.
4. Build the minimal TMA descriptor smoke for the 10 global tiles only, with
   expected mbarrier bytes equal to `81920 B/chunk`.
5. Keep `dstates` local stages and tiny scalar/vector slices on non-TMA copies.
6. If TMA still consumes exactly `131072 B`, do not add a second ping-pong tile
   until the base SMEM alias plan is reduced.
