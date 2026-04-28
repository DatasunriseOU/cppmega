# M2RNN backward launch tuning - 2026-04-28

Branch: `worker/m2rnn-prod-bwd-launch`
Base requested: `a5e7b1e`
Hardware: single `NVIDIA GB10`
Shape: `B=2,S=4096,H=44,K=64,V=16,bf16`
Path: full production mixer, `CPPMEGA_M2RNN_KERNEL=triton`

## Baseline

Command:

```bash
CPPMEGA_M2RNN_KERNEL=triton \
python tools/bench_m2rnn_kernels_nam56r.py --kernels triton --warmup 2 --iters 5
```

Result:

| fwd+bwd mean ms | stdev ms | fwd-only mean ms | peak MiB | finite |
| ---: | ---: | ---: | ---: | --- |
| 39.58 | 0.21 | 10.01 | 1712.1 | yes |

Torch profiler, one warmed iteration:

| event | CUDA total ms |
| --- | ---: |
| `_M2RNNFnBackward` | 13.94-14.59 |
| `_m2rnn_bwd_kernel` aggregate | 13.52-14.16 |
| `_m2rnn_fwd_kernel` | 4.90-5.36 |

Nsight Systems was available and a short capture was attempted:

```bash
CPPMEGA_M2RNN_KERNEL=triton \
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/m2rnn_bwd_baseline \
  python tools/bench_m2rnn_kernels_nam56r.py --kernels triton --warmup 2 --iters 1
```

That capture did not retain the full backward kernel stream on this local run,
so the tuning decision below uses bench wall time plus torch profiler evidence.

## Sweep

All rows use the same production mixer command path with `warmup=2`. The first
pass used `iters=3`; the confirmation pass used `iters=5`.

| config | fwd+bwd ms | stdev ms | fwd ms | bwd estimate ms |
| --- | ---: | ---: | ---: | ---: |
| default repeat | 39.10 | 0.20 | 10.50 | 28.60 |
| `CPPMEGA_M2RNN_BWD_NUM_WARPS=2` | 39.05 | 0.64 | 9.74 | 29.32 |
| `CPPMEGA_M2RNN_BWD_NUM_STAGES=4` | 38.44 | 0.41 | 9.72 | 28.72 |
| `CPPMEGA_M2RNN_BWD_MAXNREG=192` | 38.75 | 0.47 | 10.22 | 28.54 |
| `CPPMEGA_M2RNN_RECOMPUTE_NUM_WARPS=1` | 38.44 | 0.47 | 10.02 | 28.42 |
| `CPPMEGA_M2RNN_BWD_CHUNK_SIZE=128` | 38.82 | 0.55 | 9.71 | 29.11 |
| `CPPMEGA_M2RNN_BWD_CHUNK_SIZE=256` | 38.46 | 0.74 | 9.74 | 28.72 |

The apparent best single knobs are small wins, but they are close to measured
noise and did not compose reliably. The `BWD_NUM_STAGES=4` torch-profiler check
still showed `_m2rnn_bwd_kernel` aggregate at about `14.07 ms`, so there is no
strong evidence to change production defaults from `num_warps=4,num_stages=3`.

## Patch

Defaults are unchanged. The patch only exposes experimental env-driven launch
knobs for per-machine profiling:

```bash
CPPMEGA_M2RNN_BWD_NUM_WARPS=2
CPPMEGA_M2RNN_BWD_NUM_STAGES=4
CPPMEGA_M2RNN_BWD_MAXNREG=192
CPPMEGA_M2RNN_RECOMPUTE_NUM_WARPS=1
CPPMEGA_M2RNN_RECOMPUTE_NUM_STAGES=2
CPPMEGA_M2RNN_RECOMPUTE_MAXNREG=96
```

Recommendation for H200: rerun the same sweep there before baking in any
default. The most useful first H200 candidates are `BWD_NUM_STAGES=4`,
`RECOMPUTE_NUM_WARPS=1`, and chunk sizes `128`/`256`, but GB10 evidence is not
strong enough to recommend a default change.
