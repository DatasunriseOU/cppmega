# PR: MambaModel should use `LinearCrossEntropyModule` for `output_layer` (parity with GPTModel)

**Target**: `NVIDIA/Megatron-LM` (branch: `dev`)

## Problem

On the `dev` branch `GPTModel` constructs its `output_layer` as a
`LinearCrossEntropyModule` (subclass of `ColumnParallelLinear` that accepts
an `output_cross_entropy_loss` kwarg and, when `cross_entropy_loss_fusion=True`
and `cross_entropy_fusion_impl="linear"`, fuses the final GEMM with
cross-entropy so the `[s, b, V]` logits tensor is never materialized). The
Mamba model silently regressed and still uses a plain `ColumnParallelLinear`,
so hybrid Mamba models cannot benefit from linear-CE fusion.

### Exact locations (current `dev` @ 8a29fd5 / 9d71cb1)

`megatron/core/models/gpt/gpt_model.py`:
```
 28: from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
...
251:            self.output_layer = LinearCrossEntropyModule(
252:                config.hidden_size,
253:                self.vocab_size,
...
```

`megatron/core/models/mamba/mamba_model.py`:
```
264:            self.output_layer = tensor_parallel.ColumnParallelLinear(
265:                config.hidden_size,
266:                self.vocab_size,
...
```

There is no import of `LinearCrossEntropyModule` in `mamba_model.py` and
the `forward()` path still uses `compute_output_layer_and_language_model_loss(...)`
(the non-fused path) instead of calling
`self.output_layer(output_cross_entropy_loss=self.fuse_linear_cross_entropy, ...)`.

### History

1. **PR #3226** "[DEV] Reapply fix Linear CE Fusion" — merged to `dev` 2026-02-04 01:47 UTC.
   This PR wired `LinearCrossEntropyModule` into **both** `gpt_model.py`
   and `mamba_model.py`. The Mamba diff (from `pulls/3226/files`) was:

   ```diff
   -from megatron.core import tensor_parallel
   +from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
   ...
   +        self.fuse_linear_cross_entropy = (
   +            self.config.cross_entropy_loss_fusion
   +            and self.config.cross_entropy_fusion_impl == "linear"
   +        )
   ...
   -            self.output_layer = tensor_parallel.ColumnParallelLinear(
   +            self.output_layer = LinearCrossEntropyModule(
   ```

2. **PR #3207** "Reapply 'Add MTP support for hybrid models (#2363)'" — merged
   to `dev` 2026-02-04 22:40 UTC (~21 hours after #3226). The MTP replay
   was rebased on a pre-#3226 snapshot of `mamba_model.py` and clobbered
   the `LinearCrossEntropyModule` wiring — reintroducing plain
   `tensor_parallel.ColumnParallelLinear(...)` and dropping the
   `self.fuse_linear_cross_entropy` plumbing. The MTP replay also changed
   the `post_process` guard to `post_process or self.mtp_process`, so the
   regression affects both the main decoder head and the MTP head.

3. No open PR against `dev` fixes this (verified by searching
   `repos/NVIDIA/Megatron-LM/pulls?state=open` for titles matching
   `Mamba.*CE|Linear.*Mamba|Hybrid.*cross`, 2026-04-14).

Note: `main` has neither side using `LinearCrossEntropyModule` yet — PR #3226
is dev-only until the next sync. This PR targets `dev` directly.

## Impact

- Hybrid Mamba models cannot take the `--cross-entropy-loss-fusion
  --cross-entropy-fusion-impl linear` path even when the flag is set.
- At our NAM56R config (hidden=4096, vocab=151552, MBS=12, seqlen=4096)
  the non-fused path materializes `[s, b, V]` logits = `12288 * 151552 *
  2 B ≈ 3.6 GiB` per microbatch before the CE kernel, plus another ~3.6
  GiB for the backward grad_logits buffer — **~7 GiB unnecessary peak
  allocation per MB**. At 10 MBs per pipeline slot this is the difference
  between fitting MBS=12 and OOM-ing the 141 GiB H200 budget. We see this
  reflected in bench3 MBS=12 training crashes (documented in our internal
  `reference_main_head_liger_ce_gap.md`).
- GPT models get the fusion; Mamba models don't — silent asymmetry.

## Proposed patch

Restore the PR #3226 Mamba wiring. Minimal diff against
`megatron/core/models/mamba/mamba_model.py` on current `dev`:

```diff
--- a/megatron/core/models/mamba/mamba_model.py
+++ b/megatron/core/models/mamba/mamba_model.py
@@ -4,7 +4,6 @@
 
 from torch import Tensor
 
-from megatron.core import tensor_parallel
 from megatron.core.config_logger import has_config_logger_enabled, log_config_to_disk
 from megatron.core.inference.contexts import BaseInferenceContext
 from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
@@ -16,6 +15,7 @@
 from megatron.core.tensor_parallel import gather_from_sequence_parallel_region
 from megatron.core.transformer import TransformerConfig
 from megatron.core.transformer.enums import ModelType
+from megatron.core.transformer.linear_cross_entropy import LinearCrossEntropyModule
 from megatron.core.transformer.spec_utils import ModuleSpec, build_module
 from megatron.core.utils import (
     WrappedTensor,
@@ -NNN,6 +NNN,11 @@ def __init__(
         # TODO: remove this dependency ?
         self.model_type = ModelType.encoder_or_decoder
 
+        self.fuse_linear_cross_entropy = (
+            self.config.cross_entropy_loss_fusion
+            and self.config.cross_entropy_fusion_impl == "linear"
+        )
+
         if self.pre_process or self.mtp_process:
             self.embedding = LanguageModelEmbedding(
@@ -264,7 +269,7 @@ def __init__(
 
         # Output
         if post_process or self.mtp_process:
-            self.output_layer = tensor_parallel.ColumnParallelLinear(
+            self.output_layer = LinearCrossEntropyModule(
                 config.hidden_size,
                 self.vocab_size,
                 config=config,
```

And in `forward()`, route through the fused kernel when
`self.fuse_linear_cross_entropy` is set (mirroring `gpt_model.py`):

```diff
-        loss = self.compute_output_layer_and_language_model_loss(
-            hidden_states,
-            labels,
-            weight=self.shared_embedding_or_output_weight(),
-            sequence_parallel_enabled=self.output_layer.sequence_parallel,
-            column_parallel_linear=self.output_layer,
-            col_linear_kwargs={
-                "weight": output_weight,
-                "runtime_gather_output": runtime_gather_output,
-            },
-        )
+        output_layer_kwargs = dict(
+            input_=hidden_states,
+            weight=output_weight,
+            runtime_gather_output=runtime_gather_output,
+        )
+        if self.fuse_linear_cross_entropy:
+            loss = self.output_layer(
+                output_cross_entropy_loss=True,
+                labels=labels,
+                **output_layer_kwargs,
+            )
+        else:
+            logits, _ = self.output_layer(**output_layer_kwargs)
+            loss = self.compute_language_model_loss(labels, logits)
         return loss
```

This is exactly the diff PR #3226 originally landed; it was overwritten by
#3207 rebase-miss. We recommend restoring verbatim, then guarding MTP head
consistency in a separate commit.

## Files changed

- `megatron/core/models/mamba/mamba_model.py`

## Testing

- Reproducer: `examples/11_mamba_linear_ce/reproducer.py` constructs a
  minimal `GPTModel` and `MambaModel` side-by-side (single-rank, dummy
  `ProcessGroupCollection`), asserts
  `isinstance(gpt.output_layer, LinearCrossEntropyModule)` (passes) and
  `isinstance(mamba.output_layer, LinearCrossEntropyModule)` (fails),
  then applies the monkey-patch fix and re-asserts (passes).
- Our production workaround
  (`cppmega/megatron/apply_linear_ce_patch.py`) does the class-swap
  at runtime via `CPPMEGA_MAIN_HEAD_LINEAR_CE=1` and has been running
  on 8×H200 NAM56R MBS=10 production (269 TFLOP/s, Liger CE kernel
  routed via `_install_liger_compute`) since 2026-04-14.
- No new test files required for this PR — existing GPT linear-CE
  functional tests should extend to Mamba once the module type matches.
