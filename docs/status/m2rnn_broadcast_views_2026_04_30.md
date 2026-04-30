# M2RNN broadcast-view scan optimization - 2026-04-30

Scope: Wave9B Mamba/M2RNN scan bottleneck pass.  Worktree:
`/home/dave/source/cppmega-wave9B-m2rnn-scan-opt`.

## Context checked

- Existing cppmega Mamba/M2RNN code:
  - `cppmega/megatron/m2rnn_triton.py`
  - `cppmega/megatron/m2rnn_spec.py`
  - `cppmega/megatron/noconv_mamba_mixer.py`
- Required status docs:
  - `docs/status/mamba_ssd_scan_nsys_isolation_2026_04_27.md`
  - `docs/status/m2rnn_fwd_nsys_isolation_2026_04_27.md`
  - `docs/status/m2rnn_pararnn_parallelism_2026_04_28.md`
- Prior worktrees under `.claude/worktrees/m2rnn-*`, especially:
  - `m2rnn-pararnn` commits `6378938` and `a5e7b1e`
  - `m2rnn-prod-bwd-launch`
  - `m2rnn-prod-bwd-memory`
  - `m2rnn-tiled-{triton,cuda,tilelang}`
  - `m2rnn-apple-port`
  - `m2rnn-block-affine-proto`
- Nanochat references:
  - `../nanochat/nanochat/m2rnn.py`
  - `../nanochat/nanochat/megatron_m2rnn.py`
  - `../nanochat/docs/mamba_integration_log.md`

No web search was needed; the local prior worktrees already covered the
ParaRNN, TileLang/Triton/CUDA, and scan-kernel decision space.

## Optimization

Production M2RNN uses one q head and one k head broadcast to the common
M2RNN head count (`H=44` in the local GB10 quarter profile).  Current `main`
uses `repeat_interleave(...).contiguous()` before calling the Triton scan, so
q/k broadcasts become large materialized tensors even though the Triton kernels
already take explicit strides.

This patch keeps `1 -> H` head broadcasts as stride-0 `expand` views by default:

```text
CPPMEGA_M2RNN_BROADCAST_VIEWS=1  # default
```

Non-singleton broadcasts still materialize with `repeat_interleave`, preserving
the existing general path.  The legacy q/k gradient reduction remains the
default.  A direct atomic q/k reduction was tested and kept opt-in only:

```text
CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK=0  # default
```

It is not a default because it changes reduction order/precision.

## Direct scan microbench

Command shape:

```text
B=4 S=4096 H=44 K=64 V=16 dtype=bf16
q_heads=1 k_heads=1 v/W/xf_heads=44
CPPMEGA_M2RNN_SAVE_HNEW=0 CPPMEGA_M2RNN_BWD_CHUNK_SIZE=64
```

All runs were under `flock /tmp/cppmega_gpu_profile.lock`.

| Path | Fwd ms | Fwd+bwd ms | Peak MiB | Loss | Grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-main behavior (`BROADCAST_VIEWS=0`, q/k reduce off) | 6.034 | 53.537 | 784.4 | 16.619579 | 0.882936 |
| broadcast views only (`BROADCAST_VIEWS=1`, q/k reduce off) | 4.964 | 52.121 | 432.4 | 16.619579 | 0.882936 |
| broadcast views + direct q/k reduce (`BWD_REDUCE_BROADCAST_QK=1`) | 4.893 | 50.440 | 381.1 | 16.619579 | 0.882936 |

Defaulted result:

- fwd scan: `-1.070 ms` (`-17.7%`)
- fwd+bwd scan: `-1.416 ms` (`-2.6%`)
- direct scan peak allocation: `-352.0 MiB` (`-44.9%`)

The direct q/k reduction is faster and lower-memory in this microbench, but it
is excluded from defaults because it changes reduction order/precision.

## Real-data GB10 quarter run

Command:

```bash
ROOT=/home/dave/source/cppmega-wave9B-m2rnn-scan-opt \
  scripts/local_gb10_quarter_train.sh --train-iters 6
```

Baseline used:

```text
CPPMEGA_M2RNN_BROADCAST_VIEWS=0
CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK=0
```

Default patch used:

```text
CPPMEGA_M2RNN_BROADCAST_VIEWS=1
CPPMEGA_M2RNN_BWD_REDUCE_BROADCAST_QK=0
```

Logs:

- `/home/dave/logs/wave9B_m2rnn_disabled_20260430_003144.log`
- `/home/dave/logs/wave9B_m2rnn_broadcast_views_20260430_003853.log`

Steady-state uses iterations 3-6 to skip compile/autotune warmup.

| Path | Steady ms/iter | Tok/s | Peak max allocated MB | Iter 6 lm loss | Iter 6 grad norm |
| --- | ---: | ---: | ---: | ---: | ---: |
| current-main behavior | 4876.175 | 3360.0 | 28337.51 | 5.588483 | 49.955 |
| broadcast views default | 4880.275 | 3357.2 | 28157.83 | 5.574980 | 43.404 |

Interpretation:

- End-to-end step time is flat within local GB10 noise (`+4.1 ms`, `+0.08%`).
  The local quarter profile has one R layer, so the measured `~1.4 ms` direct
  scan fwd+bwd win is too small to move a `~4.9 s` full training step reliably.
- Peak model-step allocation drops by `179.68 MB` in Megatron's reported CUDA
  max allocation.
- No skipped or NaN iterations occurred in either run.
- Loss/grad norm stay finite.  Exact later-step values are not bitwise stable
  across repeated local runs because other Triton/FP8 paths are nondeterministic;
  the direct scan microbench preserves the tested loss/grad norm exactly for
  the default broadcast-view change.

## Validation

Passed:

```bash
python -m py_compile \
  cppmega/megatron/m2rnn_triton.py \
  cppmega/megatron/m2rnn_spec.py \
  cppmega/features/m2rnn/config.py \
  tests/test_m2rnn_triton.py \
  tests/test_m2rnn_triton_runtime_config.py \
  tests/test_m2rnn_config.py

pytest -q tests/test_m2rnn_config.py tests/test_m2rnn_triton_runtime_config.py

flock /tmp/cppmega_gpu_profile.lock -c \
  'PYTHONPATH=. pytest -q tests/test_m2rnn_triton.py -k "broadcast_single_heads or broadcast_views_can_be_disabled or bwd_reduced_broadcast_qk or bwd_broadcast_qk_value_parity or bwd_broadcast_heads or bwd_fp32"'
```

Observed:

```text
141 passed, 19 warnings
8 passed, 9 deselected, 48 warnings
```
