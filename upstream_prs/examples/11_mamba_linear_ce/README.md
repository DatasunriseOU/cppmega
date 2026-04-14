# Reproducer: `MambaModel.output_layer` vs `GPTModel.output_layer` class-parity regression

## One-line

On `NVIDIA/Megatron-LM@dev`, `GPTModel.output_layer` is a
`LinearCrossEntropyModule` (enables `--cross-entropy-loss-fusion
--cross-entropy-fusion-impl linear`), but `MambaModel.output_layer` is
plain `tensor_parallel.ColumnParallelLinear`. Hybrid Mamba models
therefore silently lose linear-CE fusion even when the flag is set.

## Upstream history

| PR | Date (UTC) | Effect on `mamba_model.py::output_layer` |
| --- | --- | --- |
| [#3226](https://github.com/NVIDIA/Megatron-LM/pull/3226) "[DEV] Reapply fix Linear CE Fusion" | 2026-02-04 01:47 | Set `LinearCrossEntropyModule` for **both** GPT and Mamba |
| [#3207](https://github.com/NVIDIA/Megatron-LM/pull/3207) "Reapply Add MTP support for hybrid models" | 2026-02-04 22:40 | Rebased on pre-#3226 snapshot, reverted Mamba side back to `ColumnParallelLinear` |

After #3207 the GPT side is untouched (still fused) but the Mamba side is
back on the non-fused path — a silent asymmetry.

As of 2026-04-14 no open PR in `NVIDIA/Megatron-LM` addresses this gap
(verified via `gh api repos/NVIDIA/Megatron-LM/pulls?state=open`
filtered on `Mamba.*CE|Linear.*Mamba|Hybrid.*cross`).

## Run

```bash
pip install -r requirements.txt
python reproducer.py
```

The reproducer initialises a 1-proc torch.distributed group
(`gloo` on CPU, `nccl` if CUDA is present), builds a minimal
`GPTModel` and `MambaModel`, inspects `type(output_layer)`, then
applies the proposed fix as a monkey-patch and re-verifies.

## Actual run (2026-04-14, bench3 H200, megatron-lm @ `/mnt/data/cppmega-root/megatron-lm`, commit state mirrors `dev` HEAD)

Captured verbatim from bench3 with `PYTHONPATH=/mnt/data/cppmega-root/megatron-lm`
pointing at a local checkout that has both PR #3226 and PR #3207 applied
(i.e. the bug-present state). Exit code **0**.

## Expected output — bug present (default on `dev` HEAD)

```
========================================================================
Output-layer class check (before fix)
------------------------------------------------------------------------
  GPTModel.output_layer    = megatron.core.transformer.linear_cross_entropy.LinearCrossEntropyModule
  MambaModel.output_layer  = megatron.core.tensor_parallel.layers.ColumnParallelLinear

  assert isinstance(gpt,   LinearCrossEntropyModule)  -> True
  assert isinstance(mamba, LinearCrossEntropyModule)  -> False (BUG)

========================================================================
Applying proposed fix (class-swap, equivalent to restoring PR #3226)
------------------------------------------------------------------------
  MambaModel.output_layer  = megatron.core.transformer.linear_cross_entropy.LinearCrossEntropyModule
  assert isinstance(mamba, LinearCrossEntropyModule)  -> True

========================================================================
VERDICT: regression CONFIRMED and fix VALIDATED.
  - GPTModel.output_layer is LinearCrossEntropyModule (PR #3226)
  - MambaModel.output_layer is plain ColumnParallelLinear (PR #3207 regression)
  - Class-swap (or restoring PR #3226's mamba_model.py diff) fixes it.
```

Exit code **0**.

## Expected output — upstream fix merged

Once a PR restores `LinearCrossEntropyModule` in `mamba_model.py::__init__`,
the script will print:

```
  [unexpected] MambaModel already uses LinearCrossEntropyModule.
  Has PR #3207's regression been fixed upstream?
```

Exit code **1** — a deliberate "please rerun and confirm" signal.

## Proposed upstream fix

See the PR template at
`upstream_prs/11_megatron_mamba_linear_ce_module.md`. Summary:

1. `from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule`.
2. Set `self.fuse_linear_cross_entropy = config.cross_entropy_loss_fusion
   and config.cross_entropy_fusion_impl == "linear"` in `__init__`.
3. Construct `self.output_layer = LinearCrossEntropyModule(...)` instead
   of `tensor_parallel.ColumnParallelLinear(...)`.
4. In `forward()`, when `self.fuse_linear_cross_entropy` is True, call
   `self.output_layer(output_cross_entropy_loss=True, labels=labels,
   input_=hidden_states, weight=output_weight, ...)` — the `gpt_model.py`
   pattern.

Exact diff: see `pulls/3226/files` for `megatron/core/models/mamba/mamba_model.py`.

## Runtime workaround (until upstream lands)

Our production workaround is
[`cppmega/megatron/apply_linear_ce_patch.py`](../../../cppmega/megatron/apply_linear_ce_patch.py)
— gated by `CPPMEGA_MAIN_HEAD_LINEAR_CE=1`. It wraps `MambaModel.__init__`
and reassigns `self.output_layer.__class__ = LinearCrossEntropyModule`
after construction. Safe because `LinearCrossEntropyModule` is a pure
subclass (only `forward()` differs); no new state to migrate.

## Impact at our config (NAM56R, bf16, MBS=12, hidden=4096, vocab=151552)

Without fusion the non-fused path materialises
`[s, b, V] ≈ 12288 * 151552 * 2 B ≈ 3.6 GiB` logits per microbatch plus
a matched grad_logits buffer on backward — ~7 GiB unnecessary peak per
pipeline slot. With the fix that GiB is never allocated (the fused
kernel streams logits → CE inside the kernel). MBS=12 transitions
from OOM to stable ~269 TFLOP/s on 8×H200 (with Liger kernel routed
via `_install_liger_compute`).
