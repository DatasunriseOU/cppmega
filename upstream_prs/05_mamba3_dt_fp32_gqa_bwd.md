# PR: Mamba3 DT fp32 cast + MIMO GQA backward branch

**Target repo:** `state-spaces/mamba` (upstream author implementation)

## Problem 1: DT dtype mismatch with Float16Module

When Megatron's `Float16Module` casts model parameters to bf16, `dt_bias` becomes bf16.
`F.softplus(dd_dt + self.dt_bias)` then returns bf16 DT. The TileLang MIMO kernel
expects fp32 DT/ADT tensors, causing either silent precision loss or dtype assertion.

### Fix

Add explicit `.to(torch.float32)` before softplus in `mamba3.py`:

```python
# Before:
DT = F.softplus(dd_dt + self.dt_bias)

# After:
DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))
```

Same fix on both scan paths (line 172 and 258).

## Problem 2: MIMO backward missing intermediate GQA branch

`mamba3_mimo_bwd.py` gradient accumulation only handles two cases:
- `G == 1` (MHA: all heads share one group)
- `G == H` (per-head: each head is its own group)

Models with intermediate grouping (e.g., ngroups=8, nheads=128 → 16 heads per group)
hit `else: raise` or produce incorrect gradients.

### Fix

Add GQA branch for `1 < G < H` where `H % G == 0`:

```python
elif H % G == 0:
    # GQA-style: reshape [B,S,R,H,N] → [B,S,R,G,H//G,N], sum over dim=4
    hpg = H // G
    dq = dq.reshape(B, S, R, G, hpg, N).sum(dim=4)
    dk = dk.reshape(B, S, R, G, hpg, N).sum(dim=4)
```

Applied in both `mamba3_mimo_bwd.py` and `mamba3_mimo_bwd_varlen.py`.

## Files Changed

- `mamba_ssm/modules/mamba3.py` — DT fp32 cast (2 lines)
- `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py` — GQA branch (~15 lines)
- `mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd_varlen.py` — GQA branch (~15 lines)

## Testing

- Verified on 8xH200: 279 TFLOP/s, 20 iterations stable, no NaN
- GQA config: ngroups=8, nheads=128 (d_inner=8192, headdim=64)
- Loss converges normally, gradients finite
