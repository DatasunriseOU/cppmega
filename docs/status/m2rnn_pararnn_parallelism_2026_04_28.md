# M2RNN ParaRNN parallelism notes - 2026-04-28

Scope: extract the useful parts of Apple ParaRNN (`apple/ml-pararnn`,
arXiv:2510.21450) and map them onto cppmega's current M2RNN Triton path.

## ParaRNN core idea

For a nonlinear recurrence

```text
h_t = f(h_{t-1}, x_t)
```

ParaRNN treats all `h_1..h_S` as unknowns in one block bi-diagonal nonlinear
system. Each Newton iteration linearizes that system, then solves the linear
recurrence

```text
delta_t = J_t delta_{t-1} + r_t
```

with a parallel reduction / prefix composition. Forward therefore costs roughly
`N_newton * O(log S)` sequence-depth if Newton converges in a small fixed number
of iterations. Backward is already linear and needs one analogous parallel
reduction.

The paper's useful engineering split is:

- `parallel`: PyTorch reference, good for formulas and convergence checks.
- `parallel_CUDA`: PyTorch Jacobian assembly plus CUDA parallel reduction.
- `parallel_FUSED`: recurrence, Jacobian assembly, Newton loop, and reduction in
  one CUDA kernel. This is the only form likely to matter for cppmega speed.

Their CUDA reduction is hierarchical:

- thread owns a tiny `chunk_size` of adjacent equations and does local Thomas
  reduction;
- warp reduces chunk tails using shuffle traffic;
- block reduces warp tails in shared memory;
- very long sequences need grid-level handoff through global memory and extra
  launches.

The important limitation: their provided CUDA structure assumes one thread can
hold at least one full equation's Jacobian and RHS in registers.

## Current cppmega M2RNN structure

Current hot path: `cppmega/megatron/m2rnn_triton.py` launches one persistent
Triton program per `(batch, head)` and loops over all `S` positions inside the
program.

Per row of the M2RNN state:

```text
x_t      = k_t[:, None] * v_t[None, :]
c_t      = tanh(h_{t-1} @ W + x_t)
h_t      = f_t * h_{t-1} + (1 - f_t) * c_t
out_t    = q_t @ h_t
```

With production-ish defaults, each head state is `[K,V] = [64,16]`.

The Jacobian is not diagonal, but it is structured:

- rows along `K` are independent;
- each row has a dense `V x V` block because of `h @ W`;
- for row-vector convention:

```text
d h_t[i] = d h_{t-1}[i] @ (f_t I + (1 - f_t) W diag(1 - c_t[i]^2))
```

So the full state Jacobian is `K` independent dense `16x16` blocks.

## Consequence

M2RNN is mathematically compatible with ParaRNN, but it is not a drop-in fit
for Apple kernels:

- Apple diagonal / 2x2 / 3x3 block-diagonal paths are too small for M2RNN's
  `V=16` block.
- Directly specializing their `RNNCellBlockDiagImpl<N=16>` means one thread
  carries `16x16` Jacobian values plus RHS/local recurrence state. That is
  already ~256 fp32 scalars before temporaries, so it will hit the register
  ceiling or collapse occupancy.
- The reduction composition also multiplies `16x16` matrices. That is much more
  work than the current sequential recurrence's per-step `h @ W`, so the only
  chance of a win is exposing enough sequence parallelism to compensate.

## Practical path

1. Validate Newton convergence for our recurrence before writing CUDA.
   Added probe:

```bash
python tools/probes/m2rnn_pararnn_newton_probe.py
python tools/probes/m2rnn_pararnn_newton_probe.py --device cuda --S 4096 --K 64 --V 16
```

This uses sequential substitution for the Newton linear solve, only to measure
whether residuals collapse in a small fixed number of iterations.

Initial probe results:

```text
cpu fp32, B=1 S=64  H=2 K=8  V=4 : residual 8.6e-1 -> 1.2e-7 after 6 Newton iterations
cpu fp32, B=1 S=128 H=2 K=16 V=16: residual 9.3e-1 -> 1.8e-7 after 6 Newton iterations
cuda bf16 input, B=1 S=256 H=2 K=16 V=16: residual 1.1e0 -> 4.2e-4 after 6 Newton iterations
```

The signs/formulas are consistent with the sequential recurrence. However,
unlike the ParaGRU/ParaLSTM result in the paper, three Newton iterations are
not obviously enough for random M2RNN inputs. Before a production kernel, test
actual trained activations and gates from NAM56R, because those may be much
more contractive than random probe tensors.

2. If convergence is good, build a small PyTorch/Triton prototype of the
   block-affine reduction for `V=16`.

   The reduction object per token/row is `(A_t, b_t)` with composition:

```text
(A_2, b_2) o (A_1, b_1) = (A_1 A_2, b_1 A_2 + b_2)    # row-vector convention
```

   This should be tested only on forward first, behind an env gate such as
   `CPPMEGA_M2RNN_PARARNN_FWD=1`.

3. Do not port Apple code verbatim into the hot path. For cppmega, a plausible
   CUDA shape is warp-per-`(B,H,K,tile)` or block-per-`(B,H,K)` where lanes
   cooperatively hold a `16x16` affine map. A one-thread-per-equation design is
   the wrong starting point for `V=16`.

4. Backward has the better algorithmic upside but higher integration risk.
   It needs the same block-affine scan with transposed Jacobians to compute
   recurrent adjoints, then a per-token local gradient kernel for `q/k/v/W/f`.

## Lower-risk architecture variant

If we are willing to change M2RNN's recurrence, constraining `state_weight` to
diagonal or small block-diagonal (`2x2` or `4x4`) makes the ParaRNN reduction
much closer to Apple's efficient path. This trades expressivity for a much more
credible speed path. Given current profiles show M2RNN forward around single
digit milliseconds after the fixed Triton launch, this variant may be a better
ROI than an exact dense-`V=16` ParaRNN solver.
