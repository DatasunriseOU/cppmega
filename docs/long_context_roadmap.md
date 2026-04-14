# Long-context roadmap for NAM56R

Strategy for scaling NAM56R training beyond 4k context without a model
redesign. Defines thresholds, switches, and the migration plan from the
current 4k baseline (289 TFLOP/s europe, 253 bench3) toward 128k.

## TL;DR

- **Sliding Window Attention (SWA)** is a runtime flag, not a weight change.
  Model weights stay compatible; only the attention mask changes.
- Keep full attention during pretrain at 4k (SWA would be a no-op anyway).
- Switch SWA **ON** when training seq_len exceeds ~**16k** tokens.
- 128k training = short **context extension phase** after full pretrain at
  4k: 500–2000 steps at low LR with SWA enabled + CP reopened.
- Target MFU at 128k is ~25–30% (memory-BW-bound regime; never matches 4k
  compute-bound baseline).

## Attention cost in NAM56R per seq_len

| seq_len | Full MLA (4 layers) | DSA topk=256 (9 layers) | Mamba3 MIMO scan (9 layers) |
|---|---|---|---|
| 4k | 256 MiB/sample | ~60 MiB | ~80 MiB |
| 16k | 4 GiB/sample | ~240 MiB | ~320 MiB |
| 32k | **16 GiB/sample** | ~480 MiB | ~640 MiB |
| 128k | **260 GiB/sample** ❌ | ~2 GiB | ~2.5 GiB |

The O(seq²) term is entirely from the **4 MLA layers**. DSA is already
sub-quadratic (O(seq × 256)). Mamba3 SSM is linear in seq.

## Switching thresholds

Recommended runtime config per seq_len range:

| seq_len range | MLA config | SW window | Notes |
|---|---|---|---|
| ≤ 8k | full attention | — | SWA would be a no-op; keep full for numerical behavior |
| 8k – 16k | full attention | — | still tractable (~4 GiB/sample, 4 layers × 16 GiB total) |
| 16k – 64k | **SWA on** | 8192 | each MLA token attends to last 8k tokens |
| 64k – 128k | **SWA on** | 8192 | + context extension fine-tune phase (see below) |
| > 128k | **SWA on** | 8192 or 16384 | + CP reopened (see memory `reference_cp_blocked_by_custom_mixers.md` — condition c triggered) |

Window=8192 chosen because: (a) covers typical local structure in code /
natural language, (b) at 4k seq it is a no-op (seq ≤ window), (c) at 16k+
it becomes active; (d) matches Mistral 7B / Mixtral SWA default.

## How to flip SWA

Our MLA layers use TE DotProductAttention which supports `window_size`.
The switch is a runtime arg, not a model surgery:

```bash
# Training at 4k (default today) — SWA off
--attention-backend fused_cudnn  # full attn

# Training at 32k+ — SWA on
--attention-backend fused_cudnn \
--attention-window-size 8192
```

Verify our MLA spec honors `window_size` when seq > window: read
`cppmega/megatron/mla_shared.py` and check the `window_size` pass-through
to `te.DotProductAttention`. If not wired, add the pass-through (trivial,
~5 LOC).

## Context extension phase

Industry standard (CodeLlama, Llama 3, Mistral, Mixtral, Gemma) — do
NOT train from scratch at 128k. It's 10-100× more expensive than 4k.

1. **Full pretrain** at 4k with SWA off — current trajectory, 289
   TFLOP/s target, optimize aggressively for MFU here.
2. **Context extension phase** — short fine-tune run at the target
   context length with:
   - SWA on (window=8192)
   - Low LR (1e-5 to 5e-6, 10× lower than pretrain final LR)
   - 500 to 2000 steps
   - Reopen CP (+multi-week port per `reference_cp_blocked_by_custom_mixers.md`)
     for activation-memory sharding, OR settle for MBS=1 + aggressive
     recompute.
3. **Optional annealing** — mix 4k and 128k sequences in the same batch
   during extension, e.g. 50/50, so the model doesn't forget short-seq
   behavior.

Expected throughput: 50-70% of the 4k baseline on the same hardware due
to memory-BW dominance (KV cache, per-token activation saves). Still
profitable because extension is a small tail (< 2% of total training
compute).

## What this means for cppmega optimization priorities

- 4k training optimizations (P1/P2/P3 SSM kernel work, fp8-param-gather,
  DualPipeV) — **fully applicable** through full pretrain (the majority
  of compute spend).
- Any optimization specifically targeting 128k activations (CP port,
  full recompute, CPU offload optimizer states) — **defer** until
  context extension phase is scoped. Premature now.
- `--attention-window-size 8192` plumbing in MLA — **add when we hit
  seq > 8k the first time**. One-liner.

## Reopening CP

When extension phase becomes real (seq > 64k), CP reopens from the
closed direction per condition (c) of `reference_cp_blocked_by_custom_mixers.md`:
*"compute mix shifts dramatically"*. At 128k the MLA/DSA activation
cost dominates and TP=2 for Mamba3 becomes viable because:

- Per-rank activation savings from seq/2 exceed the Mamba3 head-parallel
  overhead (3.2× at 4k) because at 128k the overall compute per step is
  bottlenecked on MLA/DSA, not Mamba scan (Mamba is linear → relatively
  cheaper slice).
- Rough arithmetic: at 128k MBS=1, Mamba is ~20-25% of compute (vs 34.5%
  at 4k). 3.2× slowdown on 20% = 64% → +44% total Mamba time. But CP
  saves 50% of MLA/DSA activations which dominate at 128k → enables
  MBS > 1 → GEMM fill → regains ~30% throughput → approximate wash.

TL;DR: reopen CP investigation AT 128k, not before.

## References

- `reference_cp_blocked_by_custom_mixers.md` — why CP is closed at 4k
- `project_mamba3_tp_is_net_loss.md` — TP=2 Mamba3 MIMO 3.2× slower
  (same overhead pattern as CP=2 head-parallel)
- `reference_mamba_ssm_optimization_plan.md` — P1/P2/P3 4k-scale
  optimizations
- TE DotProductAttention docs — `window_size` parameter for SWA
- Mistral 7B paper — canonical SWA reference
- CodeLlama paper — canonical context extension reference
