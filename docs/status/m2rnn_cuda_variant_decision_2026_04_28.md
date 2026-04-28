# M2RNN CUDA Variant Decision - 2026-04-28

Status: evidence
Canonical: docs/status/m2rnn_tiled_cuda_2026_04_28.md
Date: 2026-04-28
Scope: Cycle 5 CUDA/Triton variant decision for M2RNN tiled CUDA branch.

Device: `NVIDIA GB10`, capability `(12, 1)`, torch `2.13.0.dev20260417+cu132`.

Shape:

- `B=1, S=1024, H=4, K=32, V=16, tile_size=32, max_its=3, dtype=bf16`

Decision rule:

- A CUDA candidate must beat default CUDA by more than 20% on CUDA-event timing before it becomes worth production follow-up.
- Otherwise this CUDA branch remains resource/diagnostic, with Triton as the active path.
- Discard runs if `nvidia-smi` shows unrelated compute processes on the same GPU.

| Variant | CUDA event ms/iter | Wall ms/iter | Speedup vs default event | Decision |
| --- | ---: | ---: | ---: | --- |
| cuda_default | 11.256 | 11.257 | 1.00x | default |
| cuda_v16_warprow_opt_in | 17.425 | 17.425 | 0.65x | diagnostic_only |
| triton_reference | 0.500 | 0.501 | 22.50x | active_reference |

Measurement note:

- These are from the first uncontended Cycle 5 decision probe run. Later retry runs were discarded after `nvidia-smi pmon` showed concurrent GPU compute from unrelated `pretrain_mamba.py`, kernel bench, and pytest processes.

Recommendation:

- `pause_cuda_production_work_keep_resource_diagnostic_branch`

Row-block prototype decision:

- Not implemented as a kept variant in this cycle.
- A one-block-per-`(Be,tile,row)` summary row would need each row block to see the full previous `d[16]` and all rows of `M[16,16]` at every token step.
- CUDA blocks cannot synchronize or exchange shared state within a tile, so a safe row-block split would require extra global intermediate state or one kernel launch per token step.
- That would directly add either the large prefix storage this branch removed or thousands of fine-grained launches, so it is de-prioritized until a different matrix-composition strategy is designed.

Sources used:

- NVIDIA CUDA Occupancy Calculator: register and shared-memory use constrain active blocks/warps per SM. https://docs.nvidia.com/cuda/archive/11.7.1/cuda-occupancy-calculator/index.html
- CUDA C++ Programming Guide occupancy APIs and profiler occupancy guidance. https://docs.nvidia.com/cuda/archive/12.4.1/cuda-c-programming-guide/index.html
- CUDA C++ Best Practices Guide occupancy discussion. https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html
- CCCL/CUB WarpScan documentation for warp-wide scan primitives. https://nvidia.github.io/cccl/cub/api/classcub_1_1WarpScan.html

## Cycle 6 tanh.approx check

Date: 2026-04-28

Implemented an opt-in CUDA `tanh.approx.f32` path behind
`CPPMEGA_M2RNN_APPROX_TANH=1`.  Exact remains the default.

Files:

- `cppmega/megatron/cuda_ext/m2rnn_tiled_affine_scan.cu`
- `cppmega/megatron/m2rnn_pararnn_tiled_cuda.py`
- `tools/probes/m2rnn_tiled_cuda_stage_profile.py`
- `tools/probes/m2rnn_cuda_variant_decision.py`
- `tests/test_m2rnn_pararnn_tiled_cuda.py`

Finding:

- With the extension's existing `--use_fast_math` flag, CUDA `tanhf` already
  lowers to SASS `MUFU.TANH` on GB10.  `cuobjdump --dump-sass` showed
  `MUFU.TANH` in both the `UseApproxTanh=false` and `UseApproxTanh=true`
  template instantiations for `m2rnn_tile_summary_kernel` and
  `m2rnn_apply_tile_prefix_kernel`.
- The opt-in inline PTX path therefore validates the hypothesis but does not
  explain the Triton/CUDA gap.

Parity:

- `CPPMEGA_M2RNN_APPROX_TANH=1 pytest -q tests/test_m2rnn_pararnn_tiled_cuda.py -s`
  passed: 9 tests.
- Probe `B=1,S=33,H=2,K=4,V=16,tile=8,max_its=6`: max abs vs sequential
  `out=5.781650543212891e-06`, `h_final=1.7434358596801758e-06`.
- Direct exact-vs-approx on the same bf16 input: `out=0.0`, `h_final=0.0`.

ptxas:

| Kernel | Exact regs | Approx regs | Smem | Spill stores/loads |
| --- | ---: | ---: | ---: | ---: |
| `m2rnn_tile_summary_kernel` | 94 | 94 | 3456 B | 0 / 0 |
| `m2rnn_apply_tile_prefix_kernel` | 90 | 90 | 3520 B | 0 / 0 |
| `m2rnn_tile_summary_v16_warprow_kernel` | 74 | 74 | 3456 B | 0 / 0 |
| `m2rnn_apply_tile_prefix_v16_warprow_kernel` | 72 | 72 | 3520 B | 0 / 0 |
| `m2rnn_scan_tile_summaries_kernel` | 26 | n/a | 128 B | 0 / 0 |

Stage profile, diagnostic run with unrelated GPU compute present:

| Variant | Whole wall ms | Summary ms/iter | Apply ms/iter | Scan ms/iter |
| --- | ---: | ---: | ---: | ---: |
| CUDA default | 11.137 | 1.719 | 1.880 | 0.021 |
| CUDA `CPPMEGA_M2RNN_APPROX_TANH=1` | 11.215 | 1.735 | 1.881 | 0.021 |

Decision probe, same diagnostic environment:

| Variant | CUDA event ms/iter | Wall ms/iter | Note |
| --- | ---: | ---: | --- |
| `cuda_default` | 27.721 | 28.494 | Contended; slower than uncontended Cycle 5 |
| `cuda_approx_tanh_opt_in` | 27.789 | 28.566 | No speedup |
| `cuda_v16_warprow_opt_in` | 44.090 | 44.873 | Still slower |
| `cuda_v16_warprow_approx_tanh_opt_in` | 44.093 | 44.878 | No speedup |
| `triton_reference` | 0.560 | 1.338 | Contended but still much faster |

Current diagnosis:

- The fast Triton kernel is not faster because it alone has approximate tanh.
  Both paths are using hardware tanh under the CUDA extension's current compile
  flags.
- The larger gap remains layout/algorithmic: Triton keeps the full `(K,V)`
  state per `(B,H)` program and loads `W` once per program; this CUDA ParaRNN
  prototype splits by `(B,H,K,tile)`, runs summary and apply as separate
  kernels for every Newton iteration, reloads/indexes `W` and recomputes
  `k*v` in both summary and apply, and pays scan/update/layout stages.
- The Python ParaRNN code precomputes `x_proj` and `W_be`; the Triton fast path
  does not materialize those tensors, but it achieves the important part by
  keeping `W` resident in the program and forming `k*v` in registers once per
  token.  A CUDA precompute variant would add `x_proj[Be,S,V]` traffic and
  still would not fix the split summary/apply structure by itself.

Next CUDA hypothesis:

- Build a forward-style persistent CUDA kernel first: one CTA per `(B,H)` or
  a small fixed set of CTAs per `(B,H)`, keep `KxV` state resident, load `W`
  once, compute `k*v` in registers, and write `out/h_final` directly.  Only
  after matching Triton's forward structure should the ParaRNN/Newton tiled
  solve be revisited.

Additional sources:

- PTX ISA `tanh.approx.f32` is a fast FP32 approximation requiring `sm_75+`:
  https://docs.nvidia.com/cuda/hopper-tuning-guide/parallel-thread-execution/index.html
- CUDA warp shuffle functions exchange values within a warp:
  https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- CUDA Cooperative Groups grid synchronization requires cooperative launch for
  whole-grid sync:
  https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cooperative-groups.html
