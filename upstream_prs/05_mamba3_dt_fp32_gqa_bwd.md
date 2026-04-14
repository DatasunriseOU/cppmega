# PR: Mamba3 MIMO GQA backward branch (missing `1 < G < H`)

**Target repo:** `state-spaces/mamba` (upstream author implementation)

> **Note on scope.** The original reproducer pack exercises two bugs
> side-by-side — a Megatron `Float16Module` DT fp32→bf16 cast interaction
> AND this Mamba3-side missing GQA branch. Only the GQA branch is a
> `state-spaces/mamba` bug. The Megatron cast is a separate issue
> filed as [PR 16](16_megatron_float16module_mamba3_cast.md)
> (target: `NVIDIA/Megatron-LM`). See also
> `docs/upstream_bugs.md:187` for the Megatron-side writeup.

## Problem: MIMO backward missing intermediate GQA branch

`mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` (`mamba_mimo_bwd_combined`)
gradient accumulation only handles two cases:

- `G == 1` (MHA: all heads share one group)
- `G == H` (per-head: each head is its own group)

Models with intermediate grouping (e.g., ngroups=8, nheads=128 → 16 heads
per group) hit `else: raise ValueError("G value of {G} is not currently
supported!")` or produce incorrect gradients.

### Reproduction

With `nheads_qk=2, nheads=16` (so `1 < G=2 < H=16` and `H % G == 0`):

```
ValueError: G value of 2 is not currently supported!
```

Setting `G=1` or `G=H` hides the bug (the MHA and per-head branches both
exist). The pack 05 reproducer exercises the `G=2, H=16` GQA shape.

### Fix

Add GQA branch for `1 < G < H` where `H % G == 0`:

```python
elif H % G == 0:
    # GQA-style: 1 < G < H, H divisible by G.  Sum over heads_per_group.
    hpg = H // G
    # bias grads: [B, S, R, H, N] -> sum(batch, seq) -> [R, H, N] -> [H, R, N]
    # Must compute BEFORE reducing dq/dk (Q_bias has shape [H, R, N])
    dq_bias_tilelang = dq_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
    dk_bias_tilelang = dk_tilelang.sum(dim=(0, 1)).permute((1, 0, 2))
    # dq/dk: [B, S, R, H, N] -> [B, S, R, G, hpg, N] -> sum(dim=4)
    dq_tilelang = dq_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
    dk_tilelang = dk_tilelang.view(B, S, R, G, hpg, N).sum(dim=4)
    dmimo_v = dmimo_v.sum(dim=0)
    dmimo_z = dmimo_z.sum(dim=0) if dmimo_z is not None else None
    dD = dD.sum(dim=0) if dD is not None else None
```

Same fix needed in `mamba3_mimo_bwd_varlen.py`.

## Files Changed

- `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` — GQA branch (~15 lines)
- `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py` — GQA branch (~15 lines)

## Testing

- Verified on 8xH200: 279 TFLOP/s, 20 iterations stable, no NaN
- GQA config: ngroups=8, nheads=128 (d_inner=8192, headdim=64)
- Loss converges normally, gradients finite
- Reproducer at `upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd/reproducer.py`
  (stage `gqa_unpatched` raises; stage `gqa_patched` produces finite
  grads bitwise-identical across reruns)

## Not in scope (different bug, separate fix)

- **Mamba3 B/C layout `(r,g,n)` vs `(g,r,n)` latent bug**, only triggered
  at TP>1 with `ngroups>1`. Different tensors, different code path — our
  `05` patch touches only `dq`/`dk` *reduction* shape in backward.
- **Megatron `Float16Module` DT cast** — tracked separately as PR 16
  against `NVIDIA/Megatron-LM`. The reproducer in this pack covers both
  because they co-trigger in the same training config, but the fixes
  target different repos.
