# M2RNN forward Nsight isolation - 2026-04-27

Report inspected:
`/home/dave/logs/gb10_nsys_full_fixed2_20260427_173123_nsys.nsys-rep`
via the exported SQLite DB next to it.

## Profile result

The raw kernel summary makes `_m2rnn_fwd_kernel` look like the second largest
kernel in the capture:

| kernel | calls | total ms | avg ms | min ms | max ms | grid | block | regs |
| --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |
| `_m2rnn_fwd_kernel` | 391 | 2869.127 | 7.338 | 4.925 | 11.169 | `(4,44,1)` | mixed | mixed |
| `_m2rnn_bwd_chunk_kernel` | 256 | 125.996 | 0.492 | 0.472 | 0.793 | `(4,44,1)` | 128 | 255 |
| `_m2rnn_recompute_chunk_kernel` | 256 | 51.707 | 0.202 | 0.194 | 0.322 | `(4,44,1)` | 128 | 64 |

The 391 forward launches are not 391 real model forward calls. They are mostly
Triton autotune benchmark launches from the first profiled shape. Splitting by
launch shape:

| blockX | regs/thread | calls | total ms | avg ms |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 167 | 73 | 561.820 | 7.696 |
| 64 | 101 | 72 | 597.770 | 8.302 |
| 128 | 64 | 102 | 514.797 | 5.047 |
| 256 | 64 | 76 | 557.589 | 7.337 |
| 512 | 64 | 68 | 637.151 | 9.370 |

Only five block-128 launches occur after the autotune window. Their total is
about 25 ms, or about 5.0 ms per actual forward scan at
`B=4,S=4096,H=44,K=64,V=16`. The report aggregate is therefore dominated by
cold autotune replay, not steady-state M2RNN compute.

## Code findings

- The current cppmega Triton path already avoids the worst save-for-backward
  pressure: forward saves fp32 recurrent checkpoints every
  `CPPMEGA_M2RNN_BWD_CHUNK_SIZE` tokens and does not save full `h_new` unless
  `CPPMEGA_M2RNN_SAVE_HNEW=1`.
- At the profiled shape, checkpoint storage is about 44.7 MiB per forward:
  `B * (ceil(S/64)+1) * H * K * V * fp32`.
- Backward chunk recompute is visible but small in this report: recompute plus
  chunk backward totals about 178 ms across 256 launches.
- The forward kernel still performs the required sequential recurrence in one
  persistent program per `(batch, head)`. For `S=4096`, the recurrence itself is
  serial inside each program.
- The wrapper materializes broadcasted heads with `repeat_interleave` before
  launch. For default cppmega M2RNN head config this can expand q/k from one
  head to 44 heads. Avoiding those materialized broadcasts would require kernel
  support for original head counts and per-head source mapping.
- Nanochat's XMA path exists, but it is a different integration point: it calls
  the external `xma.functional.m2rnn` wrapper with fp32 q/k/v/W/f and relies on
  activation checkpointing. It is not a drop-in fix for this cppmega Triton
  profile.

## Patch

The forward kernel now defaults to a fixed launch matching the fastest observed
GB10 shape: `num_warps=4`, `num_stages=3` (`blockX=128`). Full Triton autotune
is still available with:

```bash
CPPMEGA_M2RNN_FWD_AUTOTUNE=1
```

The fixed launch can be adjusted without editing code:

```bash
CPPMEGA_M2RNN_FWD_NUM_WARPS=4 CPPMEGA_M2RNN_FWD_NUM_STAGES=3
```

## Microbench

Shape: `B=4,S=4096,H=44,K=64,V=16`, dtype bf16, GB10. Timed with CUDA events
around `m2rnn_scan_triton`.

| mode | first call ms | second call ms |
| --- | ---: | ---: |
| old autotune path (`CPPMEGA_M2RNN_FWD_AUTOTUNE=1`) | 5931.989 | 5.837 |
| fixed launch default (`CPPMEGA_M2RNN_FWD_AUTOTUNE=0`) | 369.851 | 5.328 |

The first-call number includes the default-stream gap while Triton compiles or
autotunes. The steady forward scan remains about 5 ms; the main win is removing
autotune benchmark launches from normal training/profile captures.
