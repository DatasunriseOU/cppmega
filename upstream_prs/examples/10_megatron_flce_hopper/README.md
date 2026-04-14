# Reproducer: Megatron-LM fused linear CE rejects Hopper (and every non-Blackwell)

See [`../../10_megatron_fused_linear_ce_hopper_support.md`](../../10_megatron_fused_linear_ce_hopper_support.md) for the full PR writeup.

## What this demonstrates

`megatron/core/fusions/fused_linear_cross_entropy.py` (on the `dev` branch of `NVIDIA/Megatron-LM`) contains:

```python
if cc[0] == 10:
    from .linear_cross_entropy.blackwell import entry as gpu_entry
    ...
else:
    raise ValueError(f"Unsupported architecture: {cc[0]}")
```

So enabling `cross_entropy_loss_fusion=True` with `cross_entropy_fusion_impl="linear"` on any non-Blackwell CUDA GPU (H100, H200, A100, L40, GB10, …) crashes the first training step. [Open PR #3345](https://github.com/NVIDIA/Megatron-LM/pull/3345) adds the Hopper (`cc[0]==9`) path but is still unmerged as of 2026-04-14.

## How to run

```bash
# On a Hopper/H200 machine (e.g. bench3 / europe):
source /mnt/data/venv/bin/activate          # bench3
# or
source /home/dave/cppmega-root/cppmega-venv/bin/activate   # europe

python reproducer.py
```

If `megatron-core` is the PyPI wheel (core_v0.16.x), the reproducer will exit 77 (SKIP) because that wheel doesn't yet ship the offending file. Point the env at a `dev`-branch clone to see the bug:

```bash
PYTHONPATH=/mnt/data/cppmega-root/megatron-lm python reproducer.py
```

## Expected output

### On H200 (cc=(9, 0)) — pre-fix

```
[env] platform=Linux-6.8...-x86_64-with-glibc2.39
[env] python=3.13.1
[env] torch=2.12.0.dev20260403+cu132
[env] cuda.is_available=True
[env] cuda.cc=(9, 0)  device='NVIDIA H200'  device_idx=0
[env] megatron-core=0.17.0.dev0
[reproducer] calling _get_platform() on cc=(9, 0) …
[reproducer] caught ValueError: 'Unsupported architecture: 9'
[ok] BUG REPRODUCED on cc=(9, 0). ...
```

Exit code: **0** (bug reproduced as expected).

### On Blackwell B200 (cc=(10, 0)) — works today

```
[reproducer] calling _get_platform() on cc=(10, 0) …
[ok] cc=(10, 0) (Blackwell) — native Blackwell entry loaded (expected).
```

Exit code: **0**.

### On H200 after PR #3345 merges

```
[reproducer] calling _get_platform() on cc=(9, 0) …
[ok] cc=(9, 0) (Hopper) — native Hopper entry loaded. This means PR #3345 has been applied to this tree.
```

Exit code: **0**.

Important: on our current bench3 tree this outcome only validates that the Hopper path is already patched in. It does **not** by itself prove a fresh reproduction of the original unsupported-arch bug, because the current tree is no longer unfixed.

### On macOS / CPU-only

```
[skip] no CUDA device; dispatcher asserts CUDA availability first.
```

Exit code: **77**.

## What the fix enables

Once #3345 merges (or we patch in Tier-B soft fallback), Megatron-LM users on H100/H200 can opt into the single-kernel fused LM-head path and avoid materializing the `[seq·batch, vocab]` logits tensor. In our tree this is still expected to save roughly ~6 GiB of activation memory per rank versus the materialized-logits fallback, but pack 10 should not cite `269.4 TFLOP/s` as current proof: repo source-of-truth now treats that bench3 number as superseded, and this pack still needs a retained H200 receipt before it is filing-ready.

## Local validation path (without ssh access)

If you only have a macOS / CPU machine, run the reproducer locally — it exits 77 and prints the environment. To verify the fix logic against actual CUDA hardware, either:

1. Scp the file onto bench3 (`/mnt/data/cppmega-root/cppmega/upstream_prs/examples/10_megatron_flce_hopper/`) and run there, or
2. Use any NVIDIA Megatron CI container (`nvcr.io/nvidia/pytorch:24.xx-py3`) with an H200 available, after `pip install -e git+https://github.com/NVIDIA/Megatron-LM.git@dev`.

The current bench3/europe situation should be read carefully: a patched tree can validate Hopper-path loading, but filing readiness still requires a retained H200 receipt that clearly distinguishes pre-fix reproduction from post-fix validation.
