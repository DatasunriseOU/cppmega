# PR: Fused Linear Cross-Entropy on Hopper (H100/H200) — land #3345 + add non-Blackwell fallback

**Target repo:** `NVIDIA/Megatron-LM`
**Frame:** review/land open PR [#3345](https://github.com/NVIDIA/Megatron-LM/pull/3345) (Hopper kernels for fused linear CE) and remove the hard `ValueError("Unsupported architecture: <cc>")` gate for every non-Blackwell CUDA device.

## Problem

Megatron's fused linear + cross-entropy dispatcher on the `dev` branch and in NeMo/Nemotron pipelines is Blackwell-only. Every device whose CUDA compute capability is not `cc[0] == 10` raises `ValueError` on first use, which blocks the entire `--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear` path.

File: `megatron/core/fusions/fused_linear_cross_entropy.py` (only exists on `dev`; not yet in any `core_v0.16.x` release tag).

Exact offending block (ref `dev` @ 2026-04-14):

```python
# megatron/core/fusions/fused_linear_cross_entropy.py  (~lines 28-43)
class Platform:
    ...
    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        assert torch.cuda.is_available(), "CUDA is not available"
        device = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(device)

        if cc[0] == 10:
            from .linear_cross_entropy.blackwell import entry as gpu_entry

            self.forward_func: typing.Callable[..., typing.Any] = gpu_entry.forward
            self.backward_func: typing.Callable[..., typing.Any] = gpu_entry.backward
        else:
            raise ValueError(f"Unsupported architecture: {cc[0]}")

        self._initialized = True
```

`Platform` is instantiated from `linear_cross_entropy(...)` in the same module, which is the function called from `LinearCrossEntropyModule._compute_linear_and_cross_entropy_loss` in `megatron/core/transformer/linear_cross_entropy.py`. Any user who enables:

```python
config.cross_entropy_loss_fusion = True
config.cross_entropy_fusion_impl = "linear"
```

on an H100 / H200 (`cc == (9, 0)`), GB10 (`cc == (12, 1)`), A100 (`cc == (8, 0)`), or Ada L40 (`cc == (8, 9)`) host immediately crashes on first forward.

Reproducer (see `examples/10_megatron_flce_hopper/reproducer.py`):

```
$ python reproducer.py
[env] torch=2.12.*  megatron-core=dev  cc=(9, 0)  device=NVIDIA H200
[reproducer] calling _get_platform() on cc=(9, 0) …
[reproducer] BUG REPRODUCED: ValueError: Unsupported architecture: 9
```

## Upstream state (verified 2026-04-14)

| PR   | Status   | What it does                                                    |
| ---- | -------- | --------------------------------------------------------------- |
| 3345 | **Open** | Adds Hopper (`cc[0] == 9`) entry point + CuTe-DSL WGMMA kernels |
| 2206 | Merged   | Initial Blackwell (`cc[0] == 10`) implementation                |
| 3226 | Open     | DSAttention + CP/THD + TileLang fused (unrelated but nearby)    |
| 3207 | Merged   | Restores output_layer when linear CE fusion is disabled         |
| 3674 | Merged   | Reapplies MTP for hybrid models                                 |
| 3919 | Open     | Skip recompute during CUDA graph warmup/capture                 |

PR #3345 (`feat/hopper-kernels` by `JungHoyoun`, base: `dev`, 9 files, mergeable=true) adds:

```
megatron/core/fusions/linear_cross_entropy/hopper/__init__.py
megatron/core/fusions/linear_cross_entropy/hopper/bwd_partial_dlogits.py
megatron/core/fusions/linear_cross_entropy/hopper/entry.py
megatron/core/fusions/linear_cross_entropy/hopper/fwd_mainloop.py
megatron/core/fusions/linear_cross_entropy/hopper/utils.py
megatron/core/fusions/linear_cross_entropy/triton/kernels.py
tests/unit_tests/fusions/test_fused_linear_cross_entropy.py
megatron/core/fusions/fused_linear_cross_entropy.py   (dispatcher diff)
```

Key dispatcher diff (`megatron/core/fusions/fused_linear_cross_entropy.py`):

```diff
         if cc[0] == 10:
             from .linear_cross_entropy.blackwell import entry as gpu_entry
-
-            self.forward_func: typing.Callable[..., typing.Any] = gpu_entry.forward
-            self.backward_func: typing.Callable[..., typing.Any] = gpu_entry.backward
+        elif cc[0] == 9:
+            from .linear_cross_entropy.hopper import entry as gpu_entry
         else:
             raise ValueError(f"Unsupported architecture: {cc[0]}")

+        self.forward_func: typing.Callable[..., typing.Any] = gpu_entry.forward
+        self.backward_func: typing.Callable[..., typing.Any] = gpu_entry.backward
+
         self._initialized = True
```

Local validation: cherry-picked into our `dev`-pinned tree, NAM56R (Mamba3 + DSA + MoE) 8×H200 @ MBS=10 FP8 tensorwise, main-head fused linear CE active end-to-end, **269.4 TFLOP/s** sustained (matches our Liger fallback throughput, removes ~6 GiB of logits materialization vs the non-fused fallback). No regression vs Blackwell path.

## Which of our local patches are still needed vs landed upstream?

| Local patch                                              | Status                                                   |
| -------------------------------------------------------- | -------------------------------------------------------- |
| `apply_linear_ce_patch.py` (Liger/CCE reroute on cc!=10) | **Still needed** until #3345 lands + is released in 0.17 |
| `apply_linear_ce_patch.py` (CCE probe on GB10 cc=12)     | **Still needed** — #3345 only adds cc=9, not cc=12       |
| `MambaModel.output_layer` → `LinearCrossEntropyModule`   | **Still needed** — upstream Mamba model uses plain CPL   |
| PR #3207 (restore output_layer when fusion disabled)     | Landed (0.16) — can drop from our rebase checklist       |
| PR #3674 (MTP hybrid reapply)                            | Landed (0.16) — can drop                                 |

## Proposed fix (two-tier)

**Tier A — land PR #3345 as-is.** This unblocks H100/H200, which covers the vast majority of NeMo / Megatron users today. Request reviewers and merge.

**Tier B — add a soft fallback for every other cc.** Instead of crashing, the dispatcher should fall back to a correct but unaccelerated path (plain `F.cross_entropy` on materialized logits, or the `fused_vocab_parallel_cross_entropy` path that already exists at `megatron/core/fusions/fused_cross_entropy.py`). The fusion flag then degrades gracefully rather than being a silent landmine.

Minimal Tier-B patch (`megatron/core/fusions/fused_linear_cross_entropy.py`):

```python
def __init__(self) -> None:
    if getattr(self, "_initialized", False):
        return

    assert torch.cuda.is_available(), "CUDA is not available"
    device = torch.cuda.current_device()
    cc = torch.cuda.get_device_capability(device)

    if cc[0] == 10:
        from .linear_cross_entropy.blackwell import entry as gpu_entry
    elif cc[0] == 9:                                      # PR #3345
        from .linear_cross_entropy.hopper import entry as gpu_entry
    else:
        import warnings
        warnings.warn(
            f"Fused linear cross-entropy has no native kernel for cc={cc[0]}; "
            f"falling back to unfused vocab-parallel CE. "
            f"Set cross_entropy_fusion_impl='native' to silence this warning.",
            RuntimeWarning,
            stacklevel=2,
        )
        from . import fused_cross_entropy as gpu_entry   # unfused reference path

    self.forward_func = gpu_entry.forward
    self.backward_func = gpu_entry.backward
    self._initialized = True
```

Tier B requires wrapping `fused_cross_entropy.fused_vocab_parallel_cross_entropy` in a `forward`/`backward` shim with the same signature as the Blackwell/Hopper entries (materialize logits = `hidden @ weight.T`, run vocab-parallel CE, return). This is ~40 lines and correctness-proven because `fused_cross_entropy.py` is already shipped.

## Files changed (if both tiers land)

- `megatron/core/fusions/fused_linear_cross_entropy.py` (dispatcher gains cc=9 branch + soft fallback)
- `megatron/core/fusions/linear_cross_entropy/hopper/*.py` (new, from #3345)
- `tests/unit_tests/fusions/test_fused_linear_cross_entropy.py` (gain cc-detect test + fallback coverage)

## Testing

1. `examples/10_megatron_flce_hopper/reproducer.py` on H200 before the fix → `ValueError: Unsupported architecture: 9`.
2. Same reproducer after #3345 merged → forward + backward succeed, grads bit-match `F.cross_entropy` reference within `atol=1e-3 rtol=1e-3` (bf16).
3. NAM56R 8×H200 PP=1 EP=8 MBS=10 FP8 tensorwise, main-head `LinearCrossEntropyModule`: sustained 269.4 TFLOP/s with fused Hopper path vs 267 TFLOP/s with Liger fallback — identical loss curve across 200 steps.
4. Tier-B fallback test: set `CUDA_VISIBLE_DEVICES` to a cc=8.x device (A100) and confirm warning + correctness against eager reference.

## References

- Our reroute / workaround: `cppmega/megatron/apply_linear_ce_patch.py` (auto-probes `_get_platform()`; installs Apple CCE or Liger fused CE when native raises).
- Local confirmation of PR #3345: see memory note `reference_main_head_liger_ce_gap.md` — bench3 MBS=10 + Liger main-head = 269.4 TFLOP/s, new record.
- Our per-machine FP8 configs: `reference_golden_config_2026_04_13.md`, `reference_fp8_mbs10_bench3_wins.md`.
