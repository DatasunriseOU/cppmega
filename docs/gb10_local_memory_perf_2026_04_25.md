# GB10 Local Memory And Perf Session - 2026-04-25

This note records the local single-GB10 changes and measurements for the
NAM56R-quarter debug run. The run uses real 4k clang data, FlashAttention, Muon,
bf16 no-master fallback, and quantized Muon momentum.

## Implemented In cppmega

- `cppmega/megatron/m2rnn_triton.py`: replaced forward saving of full
  `y[B,S,H,K,V]` with sparse fp32 recurrent checkpoints plus chunk recompute in
  backward. Default chunk size is `CPPMEGA_M2RNN_BWD_CHUNK_SIZE=64`.
- `tests/test_m2rnn_triton.py`: added checkpointed backward parity coverage and
  a test that the default forward allocates checkpoints, not full `y`.
- `scripts/local_gb10_quarter_train.sh`:
  - defaults `MEGATRON_BIAS_GELU_IMPL=te`;
  - keeps FlashAttention pinned with `--use-flash-attn --attention-backend flash`;
  - makes scalar fallback selectable via `CPPMEGA_MUON_SCALAR_OPTIMIZER`;
  - passes `--muon-quantized-momentum-block-size`;
  - defaults `CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER=1`;
  - adds local PyTorch profiler hooks;
  - adds an embedded `nsys profile` launch option;
  - updates the memory estimator for q8 Muon state and no-contiguous-grad local
    DDP mode.
- `docs/lion8bit_ab_2026_04_25.md`: records the Adam8bit vs Lion8bit scalar
  fallback A/B.

## Required Local Megatron Patch

These are local patches in `/home/dave/megatron-lm`. That tree is detached at
`e40feed4a` and has substantial unrelated pre-existing changes, so this session
did not commit it as a bulk Megatron change.

- `megatron/core/optimizer/emerging_optimizers.py`: uses
  `quantized_muon_momentum_update_multi_and_normalize_groups_` end-to-end for
  quantized Muon. QKV tensors get separate Q/K/V norm groups, ordinary 2D tensors
  get one group per tensor, and the lowmem Newton-Schulz path is told the input
  is already normalized so it does not repeat the Frobenius normalization.
- `megatron/core/fusions/fused_bias_gelu.py`: adds
  `MEGATRON_BIAS_GELU_IMPL={compiled,eager,te}`. The GB10 launcher defaults to
  `te` to avoid the worse Inductor path.
- `megatron/core/distributed/distributed_data_parallel.py`,
  `distributed_data_parallel_config.py`, `param_and_grad_buffer.py`, and
  `megatron/training/arguments.py`: add
  `--local-ddp-disable-contiguous-grad-buffer` for local single-process tests.
  This bypasses Megatron's persistent contiguous DDP grad buffer and leaves
  gradients in per-parameter `.grad`.

## M2RNN Memory Win

Old forward save for a full NAM56R-quarter M2RNN layer at
`B=4,S=4096,H=44,K=64,V=16`:

```text
full y[B,S,H,K,V] bf16: 1.375 GiB
```

New default checkpoint/recompute mode:

```text
fp32 checkpoints every 64 tokens: 44.69 MiB
checkpoints plus one live chunk in backward: 89.38 MiB
```

The MBS=4 torch profile shows `_M2RNNFnBackward` at about `45 ms` CUDA total,
with `_m2rnn_bwd_chunk_kernel` at about `31 ms` across 64 launches. It is no
longer the main peak-memory source in this local run.

## DDP Grad Buffer Question

Megatron's normal DDP buffer is not currently an FP8 buffer knob. In the standard
DDP path, `grad_data` is `torch.float` if `grad_reduce_in_fp32` is set, otherwise
it is `param.dtype`; with bf16 model params that means BF16. Existing FP8/MXFP8
paths can reuse grad-buffer storage for parameter all-gather, but weight grads
are still BF16/FP32.

Making that buffer FP8 would require a separate quantized-gradient contract:
scale metadata, quantized accumulation/reduction, dequant or quantized optimizer
consumption, and safe microbatch accumulation semantics. That is not present in
this local stack or upstream Megatron's ordinary DDP buffer path. For the
single-GB10 local debug case, the cleaner win was to remove the persistent DDP
buffer entirely with `--local-ddp-disable-contiguous-grad-buffer`.

## Measurements

Baseline before this session, MBS=4 local run:

```text
after setup:        alloc 6.832 GiB, reserved 7.609 GiB
step 2 post:        alloc 11.448 GiB, reserved 31.740 GiB
step 2 max_alloc:   29.224 GiB
```

Current MBS=4 smoke with no contiguous DDP grad buffer:

```text
log: /home/dave/logs/gb10_quarter_no_contig_grad_20260425_105846.log
after setup:        alloc 3.422 GiB, reserved 4.199 GiB
step 1 post:        alloc 11.504 GiB, reserved 24.367 GiB
step 1 max_alloc:   22.932 GiB
loss:               lm 11.65463, mtp_1 11.64792
grad norm:          76.923
skipped/nan:        0/0
```

Current MBS=4 with memory hooks plus PyTorch profiler:

```text
log:   /home/dave/logs/gb10_quarter_mbs4_profile_20260425_110533.log
trace: /home/dave/logs/gb10_quarter_mbs4_profile_20260425_110533_torch_profile/train_step_2.json
table: /home/dave/logs/gb10_quarter_mbs4_profile_20260425_110533_torch_profile/train_step_2_cuda_table.txt

after setup:        alloc 3.422 GiB, reserved 4.199 GiB
step 2 post:        alloc 11.505 GiB, reserved 26.408 GiB
step 2 max_alloc:   25.095 GiB
step 2 loss:        lm 10.97315, mtp_1 11.37682
step 2 grad norm:   113.327
skipped/nan:        0/0
```

Compared with the earlier baseline, current MBS=4 reduced setup allocation by
about `3.41 GiB`, step-2 max allocation by about `4.13 GiB`, and step-2 reserved
memory by about `5.33 GiB`.

## Profiler Notes

PyTorch profiler was the reliable end-to-end capture in this session. The
embedded `nsys` path produced reports but still missed part of the child CUDA
timeline under `torch.distributed.run`.

Useful `nsys` artifacts:

```text
/home/dave/logs/gb10_quarter_mbs4_nsys_fork_20260425_111137_nsys.nsys-rep
/home/dave/logs/gb10_quarter_mbs4_nsys_fork_20260425_111137_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
```

Top PyTorch-profiler CUDA totals for step 2:

```text
TensorParallelMuon.step:                         2.615 s
qmuon_update_multi_kernel:                       40.6 ms
LinearCrossEntropyFunctionBackward/cce bwd:      ~789 ms
aten::addmm:                                     1.786 s total, 840.9 ms self
aten::mm:                                        666.8 ms total, 441.7 ms self
FlashAttnFuncBackward:                           101.0 ms
_M2RNNFnBackward:                                45.2 ms
AdamW8bit fallback step:                         28.5 ms
Memcpy DtoD:                                     124.3 ms
```

Partially useful `nsys` kernel summary still points at MoE token movement:

```text
_permute_kernel:                704.9 ms
_sort_chunks_by_map_kernel:     481.9 ms
```

Treat those `nsys` numbers as directional until we get a cleaner child-process
capture or profile the training process without the `torch.distributed.run`
wrapper.

## Next Bottlenecks

1. Make the Megatron patch durable as a clean branch or patch stack instead of
   relying on the current detached dirty tree.
2. Continue fusing the Muon path: grouped q8 update plus normalization is wired,
   but the step is still dominated by surrounding Newton-Schulz/optimizer work.
3. Improve MoE token movement and sort/permute kernels; `nsys` still shows large
   time there.
4. Inspect CCE backward and GEMM shapes from the torch trace.
5. Run a longer Lion8bit scalar fallback A/B before changing the default.
