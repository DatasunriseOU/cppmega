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
