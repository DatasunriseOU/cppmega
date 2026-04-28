# M2RNN Backward Memory/Recompute Check - 2026-04-28

Branch: `worker/m2rnn-prod-bwd-memory`
Base: `a5e7b1e990057f2cbfff1693e3cb6707cf13cb94`
Hardware: local GB10, bf16, Triton kernel.

## Commands

Full production mixer baseline:

```bash
CPPMEGA_M2RNN_KERNEL=triton CPPMEGA_M2RNN_SAVE_HNEW=0 CPPMEGA_M2RNN_BWD_CHUNK_SIZE=64 \
  python tools/bench_m2rnn_kernels_nam56r.py \
  --kernels triton --batch 2 --seq 4096 --hidden 3520 --num-heads 44 --warmup 1 --iters 3
```

Direct recurrent scan memory/recompute sweep, matching production head topology
(`q/k` one-head, `v/xf/W` 44-head):

```bash
python tools/profiling/bench_m2rnn_backward_memory.py \
  --batch 2 --seq 4096 --heads 44 --q-heads 1 --k-heads 1 \
  --chunk-sizes 32 64 128 --save-hnew 0 1 --warmup 1 --iters 5
```

Torch profiler tables:

```bash
python tools/profiling/bench_m2rnn_backward_memory.py \
  --batch 2 --seq 4096 --heads 44 --q-heads 1 --k-heads 1 \
  --chunk-sizes 64 --save-hnew 0 --warmup 1 --iters 2 \
  --profile --profile-save-hnew 0 --profile-chunk-size 64 \
  --profile-rows 40 --profile-out /tmp/m2rnn_recompute_chunk64_prof.txt

python tools/profiling/bench_m2rnn_backward_memory.py \
  --batch 2 --seq 4096 --heads 44 --q-heads 1 --k-heads 1 \
  --chunk-sizes 32 --save-hnew 1 --warmup 1 --iters 2 \
  --profile --profile-save-hnew 1 --profile-chunk-size 32 \
  --profile-rows 30 --profile-out /tmp/m2rnn_savehnew_chunk32_prof.txt
```

## Results

Full mixer baseline:

| mode | fwd ms | fwd+bwd ms | peak GPU MiB | finite |
| --- | ---: | ---: | ---: | --- |
| `CPPMEGA_M2RNN_KERNEL=triton`, `SAVE_HNEW=0`, chunk 64 | 10.14 | 38.63 +/- 0.78 | 1713.1 | yes |

Direct recurrent scan:

| save | chunk | fwd ms | fwd+bwd ms | stdev | peak alloc MiB | active peak MiB | recurrent MiB | finite |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 32 | 4.17 | 26.40 | 0.30 | 180.2 | 180.2 | 55.8 | yes |
| 0 | 64 | 4.27 | 27.44 | 0.24 | 158.2 | 158.2 | 44.8 | yes |
| 0 | 128 | 4.20 | 27.35 | 0.45 | 147.9 | 147.9 | 55.8 | yes |
| 1 | 32 | 4.75 | 25.96 | 0.24 | 884.2 | 884.2 | 759.8 | yes |
| 1 | 64 | 4.84 | 27.24 | 0.54 | 862.2 | 862.2 | 748.8 | yes |
| 1 | 128 | 4.71 | 27.68 | 0.50 | 851.9 | 851.9 | 759.8 | yes |

Profiler highlights:

| variant | `_m2rnn_fwd_kernel` | `_m2rnn_recompute_chunk_kernel` | `_m2rnn_bwd_chunk_kernel` | `_M2RNNFnBackward` CUDA total |
| --- | ---: | ---: | ---: | ---: |
| save 0, chunk 64 | 3.960 ms | 6.979 ms / 64 calls | 15.151 ms / 64 calls | 22.203 ms |
| save 1, chunk 32 | 4.585 ms | 5.450 ms / 128 calls | 13.981 ms / 128 calls | 19.487 ms |

## Findings

- `SAVE_HNEW=1` is not a production default win for this shape. The best GB10
  timing in this sweep is only 0.44 ms faster than recompute (`25.96` vs
  `26.40` ms), while peak allocation increases by about 704 MiB.
- Saved `h_new` does not remove `y_chunk` recompute. Backward still launches
  `_m2rnn_recompute_chunk_kernel` to reconstruct gated states for the local
  reverse sweep. `SAVE_HNEW=1` only removes the candidate recompute inside
  `_m2rnn_bwd_chunk_kernel`.
- Chunk 64 remains the best memory/perf compromise for the default path:
  lower recurrent allocation than chunk 32/128 with no meaningful timing loss
  at full-mixer scale.
- `q/k` expanded temporaries are already avoided in production: one-head
  broadcasts stay as stride-0 views and `dq/dk` reduce directly into one-head
  fp32 buffers when `CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK=1` (default).
- Remaining removable allocations are not worth a kernel change without a
  larger redesign:
  - `v` and `xf` are real 44-head production tensors, not removable expanded
    temps.
  - `dW_slabs` is tiny at this shape (`B*H*V*V*fp32`, about 88 KiB for B=2);
    replacing it with atomics would save little and risks nondeterministic
    accumulation noise.
  - `h_new_save` dummy when disabled is a one-element tensor; measurable impact
    is below the profiler noise floor.
  - `y_chunk` is required by the current two-kernel chunked backward. Avoiding
    it would require a fused recompute+reverse algorithm or much denser
    checkpoints, not a safe local temp cleanup.

## Recommendation

Keep `CPPMEGA_M2RNN_SAVE_HNEW=0` and `CPPMEGA_M2RNN_BWD_CHUNK_SIZE=64` as the
production defaults. For H200, the hypothesis is also to keep recompute: H200
should reduce the small candidate recompute cost faster than it reduces the
activation memory pressure, while `SAVE_HNEW=1` still adds a full
`B*S*H*K*V` bf16 tensor per M2RNN layer. If H200 is tested, treat
`SAVE_HNEW=1, chunk=32` as an opt-in experiment only and require a steady-state
nsys profile before considering it.
