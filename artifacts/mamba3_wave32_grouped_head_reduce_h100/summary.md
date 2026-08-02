# Wave32 grouped_head_reduce — verdict: closed as dead end

Backlog: P090 (`docs/backlog_plan_2026_08_01.md`).
Data: `wave32_grouped_head_reduce_h100_final_20260430/report.json` (H100,
warmup 20 / iters 100, torch 2.13.0.dev20260426+cu132).
Harness: `scripts/modal_mamba3_wave32_grouped_head_reduce_h100.py`.
Candidates: `cppmega/megatron/mamba3_grouped_head_reduce.py`
(`reduce_grouped_heads_torch` vs `reduce_grouped_heads_triton`).

## Question

The grouped-head backward tail reduction `[B, S, R, H, N] → [B, S, R, G, N]`
(sum over `heads_per_group`) is currently a torch `view(...).sum(dim=4)`.
Could a fused Triton kernel get within 2× of torch (or beat it)?

## Bench numbers (H100)

| shape | hpg | input pair | torch ms | triton ms | triton slowdown | torch GiB/s | triton GiB/s | max_abs dq/dk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| smoke_hpg2 (B1 S512 H16 G8 R4 N64) | 2 | 8 MiB | 0.0247 | 4.0627 | 164× | 475 | 2.9 | 0 / 0 |
| half_seq_hpg16 (B2 S2048 H128) | 16 | 512 MiB | 0.2077 | 4.1066 | 19.8× | 2557 | 129 | 0 / 0 |
| fullish_seq4096_hpg16 (B2 S4096 H128) | 16 | 1024 MiB | 0.3992 | 4.1928 | 10.5× | 2662 | 253 | 3.6e-07 / 1.2e-07 |

Numerical correctness of the Triton path is proven (max_abs ≤ 3.6e-07, bf16
noise). Performance is not closeable:

- The op is a pure streaming reduce: read 2×input, write 2×output, zero
  reuse. Torch already sustains ~2.6 TB/s on the real shapes — ~80 % of
  H100 HBM3 peak (~3.35 TB/s). A *perfect* kernel could beat torch by at
  most ~1.26×.
- The Triton candidate runs at 2.9–253 GiB/s and is nearly flat (~4.1 ms)
  across a 128× input-size range — launch/occupancy-bound, not
  bandwidth-bound. Reaching even 2× of torch needs ≥1.3 TB/s from it, i.e.
  a 5×–450× improvement over what was measured.
- Closing that gap means a from-scratch CUDA kernel (the stated alternative
  in P090) whose best-case prize is ≤1.26× over a two-line torch reduction
  that is already at 80 % of memory peak. Negative expected value.

## Verdict

**Closed as dead end.** No further Triton/CUDA work on this reduction.
Replacement (unchanged production path): the torch view+sum pair
`reduce_grouped_heads_torch` in
`cppmega/megatron/mamba3_grouped_head_reduce.py:44`
(`dq_expanded.view(B, S, R, G, hpg, N).sum(dim=4)`), already the default
consumed by the grouped-head backward patch lane
(`cppmega/megatron/upstream_patches/apply_mamba3_grouped_head_bwd_patches.py`).

Revisit only if the shape regime changes fundamentally (e.g. hpg ≫ 16 with
N > 128 making the reduce compute-bound, where fusion could pay).
