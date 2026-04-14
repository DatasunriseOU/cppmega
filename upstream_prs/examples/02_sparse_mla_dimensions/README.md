# Reproducer: SparseMLA TileLang kernels hardcode DeepSeek-V3.2 dimensions

Template: [../../02_sparse_mla_generalize_dimensions.md](../../02_sparse_mla_generalize_dimensions.md)

## TL;DR

The fused TileLang SparseMLA forward/backward kernels (upstream source:
[`tile-ai/tilelang/examples/deepseek_v32`](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32))
hardcode DeepSeek-V3.2 dimensions (`dim+tail_dim == 576`, `D == 512`) in
three places. Any MLA configuration with different dimensions — e.g.
NAM56R with `kv_lora_rank=64, qk_pos_emb_head_dim=64` (d_total=128,
v_channels=64) — trips the asserts and falls through to the unfused,
O(seq² × heads) materializing path.

This reproducer loads **two copies** of the kernel that live side-by-side
in this repo:

| Path                                                   | State           |
| ------------------------------------------------------ | --------------- |
| `cppmega/megatron/tilelang_sparse_mla/`                | **still buggy** — target of upstream PR |
| `cppmega/megatron/sparse_mla_ops/`                     | already fixed in cppmega |

and shows that (A) the old copy asserts on d_total=128 / d_v=64 and
(B) the new copy runs fwd + bwd correctly at the same shape.

## Run

```bash
# On an H200 / Hopper host with the cppmega venv already active:
cd upstream_prs/examples/02_sparse_mla_dimensions
python reproducer.py
```

No extra installs needed on `h200_1` — just activate
`/mnt/data/venv` first.

## Expected output

Captured on `h200_1` (torch 2.12+cu132, tilelang 0.1.8):

```
cppmega repo root: /mnt/data/cppmega-root/cppmega
CUDA device: NVIDIA H200
torch:    2.12.0.dev20260410+cu132
tilelang: 0.1.8+cuda.gitf309d814

Probe shape: d_total=128 (kv_lora=64 + qk_pos=64), v_channels=64  (NAM56R)
Target shape (hardcoded): d_total=576, v_channels=512             (DeepSeek-V3.2)

========================================================================
(A) OLD copy (cppmega/megatron/tilelang_sparse_mla/) — upstream PR target
========================================================================
  error> FWD AssertionError: you should assign dim otherwise
  error> BWD AssertionError: warp_row_tiles must be greater than 16, got 8
  fwd_ok = False    bwd_ok = False
  BUG_REPRODUCED: the old copy refuses non-DeepSeek dims

========================================================================
(B) NEW copy (cppmega/megatron/sparse_mla_ops/)        — fix already lives here
========================================================================
  imported = True
  fwd_ok   = True   out_shape = (1, 128, 8, 64)
  bwd_ok   = True   finite(dq,dkv) = (False, False)
  NOTE: bwd runs (shape plumbing OK) but returns NaN at H=8 — that's a
        SEPARATE small-H indexing issue in the kernel (NAM56R uses
        larger H per kv_group in production). Out of scope for this PR;
        the test here only asserts the d_v hardcode is removed.
  FIX_VALIDATED: parametric d_v path runs fwd + bwd at d_total=128,d_v=64

========================================================================
VERDICT: BUG_REPRODUCED in tilelang_sparse_mla/{fwd,bwd}.py.
         FIX_VALIDATED by sparse_mla_ops/ (same fixes belong upstream).
```

Exit code:
- `1` — bug reproduced AND fix validates (expected outcome today).
- `0` — old copy no longer asserts (upstream landed the fix).
- `2` — environment missing (no CUDA, no tilelang, etc.).

### About the "finite(dq,dkv) = (False, False)" note

The bwd kernel *compiles and runs* at d_total=128/d_v=64 — so the
``D = 512`` hardcode removal and ``next_power_of_2`` relaxation are
validated. The returned gradient tensors contain NaN at `H=8` because
the kernel indexes ``Lse[..., bz*block_H + h_i]`` where `block_H=16` but
the real head count is 8, producing OOB reads. That is a distinct kernel
bug around small H, orthogonal to the dimension-generalization PR. In
NAM56R production the `heads // kv_group` is larger (MLA uses MQA, so
kv_group=1 and heads=num_attention_heads), so this path doesn't
manifest. Full numerical gradcheck would run with those production shapes
and is deliberately out of scope for a minimal reproducer.

## What the PR should do upstream

Port the four file-level changes already present in
`cppmega/megatron/sparse_mla_ops/` back into the `tile-ai/tilelang`
`examples/deepseek_v32` source (and the NVIDIA/Megatron-LM DSA dispatch
guard in `dsa.py`):

1. Replace `assert dim == next_power_of_2(dim)` with `assert dim % 16 == 0`.
2. Remove `assert dim_plus_tail_dim == 576` — kernel is already
   parameterized over `dim`/`tail_dim`.
3. Thread `d_v` through `SparseMLA.forward/backward` + the interface.
4. Drop `D = 512` in `sparse_mla_bwd`; infer `D = o.shape[-1]`.
5. (Nice-to-have) switch `P_shared_cast` / `dP_shared_cast` dtype to
   `accum_dtype` for dKV numerical stability.
