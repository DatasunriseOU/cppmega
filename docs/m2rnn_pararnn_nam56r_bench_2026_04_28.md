# M²RNN ParaRNN end-to-end fwd+bwd bench at NAM56R dims

**Date:** 2026-04-28
**Branch:** `m2rnn-pararnn`
**Hardware:** NVIDIA GB10
**Bench script:** `tools/bench_m2rnn_kernels_nam56r.py`

## Configuration

| | |
|---|---|
| Wrapper | `CppMegaM2RNNMixer` (full mixer, including projections + conv1d + g_norm + output projection) |
| Batch | 2 |
| Sequence | 4096 |
| Hidden | 3520 |
| Heads | 44 (k_head_dim=64, v_head_dim=16) |
| Forward dtype | bf16 |
| Iters | 2 warmup + 5 timed |
| Loss | `out.float().pow(2).mean()` |

The `torch` Python-loop kernel was skipped — it's known to be ~460× slower than the sequential Triton kernel and would have taken hours per iter at this shape; the comparison that matters is the production Triton path vs the new ParaRNN path.

## Results

| kernel  | fwd+bwd ms (mean) | ± stdev | fwd only (ms) | peak GPU (MiB) | out finite | grad finite | max\|grad − grad_triton\| (input_proj) |
|---------|-------------------|---------|---------------|----------------|------------|-------------|----------------------------------------|
| triton  | 47.35             | 0.50    | 9.55          | 1036.9         | yes        | yes         | —                                      |
| pararnn | 31 362.42         | 293.74  | 27 533.76     | 19 561.2       | yes        | yes         | 4.58e-04                               |

**Pararnn is 662× slower than the sequential Triton kernel and uses 19× more GPU memory at the production NAM56R shape.** Gradient parity (4.6e-4 max abs on `input_projection.weight` in bf16) is at the expected fp32→bf16 roundoff level — the IFT backward computes the right gradient.

## Interpretation

This is the expected result, and it's why we ran the bench rather than assuming. At NAM56R the existing sequential Triton kernel already maintains B·H = 88 chains in parallel (or 176 at MBS=4), which is enough to saturate GB10's 32 SMs. The recurrence is fundamentally serial in the sequence dimension, but with enough cross-(batch, head) parallelism that wall-clock time is dominated by per-step compute, not by serial depth.

Pararnn's value is exposing log-depth parallelism *within* the sequence dimension, at the cost of building (B·H·k_dim, S, V, V) Jacobians and running ~8 Newton iterations per forward. On NAM56R that overhead is enormous: V=16 makes per-Jacobian-block work small, but the volume of those blocks (Be · S · V² ≈ 11k · 4k · 256 = 11 G fp32 elements per Jacobian) is what blows up. The Triton parallel scan kernel from Phase B.2 helps inside the chunk, but the Newton outer loop's fundamental work is still O(max_its · S · Be · V²).

## Recommendation

**Keep the sequential Triton kernel as the default for the production NAM56R recipe.** Do not switch the default in `m2rnn_spec.py`.

The `CPPMEGA_M2RNN_KERNEL=pararnn` knob stays in place as a research path. ParaRNN may pay off in regimes the production recipe doesn't currently cover:

- **Very small B·H.** When fewer than ~32 chains are available, the sequential kernel underutilises SMs and pararnn's intra-sequence parallelism could close the gap. NAM56R's MBS≥1 with H=44 already exceeds that threshold, so this regime is academic for our setup.
- **Very long sequences (S ≫ 4096) at moderate B·H.** Asymptotically the sequential kernel is Θ(S) serial depth per chain while pararnn is Θ(log S); the constant factor flips somewhere, but probably not until S is in the tens of thousands and B·H is small enough that the per-chain serial latency dominates. We don't have a workload like this in the cppmega roadmap right now.
- **Inference with very small batch.** Same logic — sub-32 chains underutilise the sequential kernel.

Phase A/B/C still earned their keep:

- Phase A: clean PyTorch reference + Newton convergence / gradient-parity tests that catch regressions in the sequential kernel too.
- Phase B.2: Triton residual+Jacobian and Brent-Kung scan kernels that deliver 3–6× over the PyTorch path *on the pararnn forward* — these would be reused if we ever ship pararnn for a sub-saturating-B·H workload.
- Phase C: IFT-as-`autograd.Function` is the right structural pattern for any future fixed-point solver in cppmega; it's also 14× faster than the autograd-through-Newton-loop torch path it replaced (see commit `87271a4` for that bench).

## Followups (optional, not blocking)

- If we ever care about the small-batch / long-S regime, repeat this bench at e.g. B=1, H=4, S=8192–32768 to find the crossover point.
- The pararnn IFT path may be a useful tool for analytic gradient-checking the sequential kernel — fp64 reference forward + fp64 IFT backward is the only path that gives machine-precision parity for `dW`, `dk`, `dv` at long S without sequential gradient checkpointing eating memory.
