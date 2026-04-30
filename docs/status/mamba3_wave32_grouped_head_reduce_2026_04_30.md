# Mamba3 Wave32 Lane C: grouped-head backward reduction

This lane benchmarks the post-TileLang grouped-head tail reduction, not
attention. The default production path remains PyTorch
`view(...).sum(dim=4)`. The new helper is default-off and selected only with
`CPPMEGA_MAMBA3_GROUPED_HEAD_REDUCE_BACKEND=triton` or an explicit Python
`backend="triton"` call.

## Scope

- Compare current PyTorch pair reduction:
  `[B, S, R, H, N] -> [B, S, R, G, N]` for both `dq` and `dk`.
- Prototype a fused Triton pair-reduction helper.
- Run H100-only component perf/correctness.
- Do not mutate production Mamba3 TileLang source in this lane.

## Commands

```bash
python -m py_compile \
  cppmega/megatron/mamba3_grouped_head_reduce.py \
  scripts/modal_mamba3_wave32_grouped_head_reduce_h100.py
pytest -q tests/test_mamba3_grouped_head_reduce.py tests/test_mamba3_grouped_head_bwd_applier.py
modal run scripts/modal_mamba3_wave32_grouped_head_reduce_h100.py::main \
  --run-id wave32_grouped_head_reduce_h100_final_20260430 \
  --warmup 20 \
  --iters 100 \
  --include-full
modal volume get cppmega-mamba3-benchmarks \
  /benchmarks/mamba3_wave32_grouped_head_reduce_h100/wave32_grouped_head_reduce_h100_final_20260430 \
  artifacts/mamba3_wave32_grouped_head_reduce_h100/
```

## Notes

The helper can reduce launch count for the pair reduction, but it cannot remove
the expanded TileLang `dq/dk` outputs by itself. A real production win still
requires the TileLang backward kernel to write original grouped heads directly
or to fuse this reduction into the owning backward epilogue.

## H100 result

Final run: `wave32_grouped_head_reduce_h100_final_20260430` on
`NVIDIA H100 80GB HBM3`, CUDA 13.2, PyTorch `2.13.0.dev20260426+cu132`.

Apps started by this lane:

- `ap-qxibyCtoYPKnE4g3PH9FrW` - stopped, `0` tasks, naive Triton probe.
- `ap-0SUEmv54wLEGRIcqCZx8sm` - stopped, `0` tasks, tiled Triton probe.
- `ap-PCPQVE1s2GWTNznAdcMdm0` - stopped, `0` tasks, final tiled run.

| shape | torch pair | Triton fused pair | speedup | torch peak | Triton peak | max_abs dq/dk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke_hpg2` | `0.024692 ms` | `4.062719 ms` | `0.006x` | `8.00 MiB` | `8.00 MiB` | `0/0` |
| `half_seq_hpg16` | `0.207740 ms` | `4.106624 ms` | `0.051x` | `64.00 MiB` | `64.00 MiB` | `0/0` |
| `fullish_seq4096_hpg16` | `0.399202 ms` | `4.192784 ms` | `0.095x` | `128.00 MiB` | `128.00 MiB` | `3.57628e-07/1.19209e-07` |

The current PyTorch `view(...).sum(dim=4)` pair reduction is already cheap:
`0.399 ms` on the full-ish component shape while reading/writing about
`1.0625 GiB` of logical data per iteration. It allocates only the two expected
outputs (`64 MiB` pair) with allocator peak delta `128 MiB`; the fused Triton
helper did not lower peak memory and was much slower.

Judgment: do not integrate the Triton helper into production. Keep the branch as
negative evidence and a reusable H100 component harness. The useful next step is
not a standalone post-kernel helper; it is a deeper TileLang/CUDA rewrite that
never materializes expanded grouped-head `dq/dk` in the first place.
