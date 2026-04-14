# PR 16: Megatron `Float16Module` silently casts Mamba3 fp32 bias/D/dt tensors to bf16

**Target repo:** `NVIDIA/Megatron-LM`

**Relates to:** `docs/upstream_bugs.md:187` (internal tracking),
pack 05 reproducer (`upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd/`)
which exercises the DT bf16 symptom alongside a separate Mamba3-side
GQA branch bug.

## Summary

`megatron/core/transformer/module.py::Float16Module.__init__` walks every
parameter of the wrapped module and converts it to bf16 (or fp16)
indiscriminately. Upstream `mamba_ssm.modules.mamba3.Mamba3` deliberately
keeps several parameters in fp32 because the TileLang scan kernel's
dispatch signature requires them in fp32:

- `Q_BIAS`
- `K_BIAS` (which is `C_bias`, `B_bias`)
- `D`
- `dt_bias`
- `mimo_x_bias`, `mimo_z_bias`, `mimo_o_bias` (MIMO paths only)

`Float16Module`'s bf16 wrapper doesn't know about this Mamba3-specific
fp32 contract and silently casts them.

## How it surfaces

Most visible symptom: `mamba_ssm.modules.mamba3.Mamba3.forward` computes

```python
DT = F.softplus(dd_dt + self.dt_bias)   # dt_bias now bf16 -> DT bf16
```

and hands `DT` to the TileLang `mamba_mimo_fwd_kernel` which declares
`DT: T.Tensor([B, H, S], T.float32)` (fp32 only). On the stacks we've
tested this produces either:

1. Low-level TVM-FFI dtype error:
   `RuntimeError: kernel mamba_mimo_fwd_kernel input DA_CS dtype expected
   float32, but got bfloat16`, OR
2. Silent numerical garbage (order of argument validation is not
   guaranteed), surfacing as `grad_norm=NaN` on iter 1 of training.

D and the other fp32-contract parameters fail in analogous ways on other
kernel dispatch paths (varlen, MIMO).

## Reproducer

The combined reproducer at
[`examples/05_mamba3_dt_fp32_gqa_bwd/reproducer.py`](examples/05_mamba3_dt_fp32_gqa_bwd/reproducer.py)
exercises both this Megatron cast and the Mamba3-side GQA branch bug
(pack 05). The DT-cast symptom corresponds to the `bf16` stage:

```
Subprocess stage: bf16
[stage bf16] Building inputs with dt_dtype=bfloat16...
[stage bf16] RESULT: kernel refused bf16 DT with:
    RuntimeError: kernel mamba_mimo_fwd_kernel input DA_CS dtype expected
    float32, but got bfloat16
STAGE_RESULT=bf16_refused
```

The `fp32` stage confirms that once DT is fp32 the kernel accepts the
call and the fwd+bwd produces finite gradients. In a Megatron+Mamba3
training loop, the fp32 path is what the Mamba3 author *intended* but
what `Float16Module` blocks.

## Current workaround in cppmega

The old per-forward pre-hook is no longer our live workaround. Current
cppmega restores these parameters exactly once inside a patched
`Float16Module.__init__`, immediately after Megatron's blanket bf16 cast:

```python
_MIMO_FP32_PARAMS = (
    "dt_bias", "D",
    "B_bias", "C_bias",
    "mimo_x", "mimo_z", "mimo_o",
)

_orig_f16_init = Float16Module.__init__

def _cppmega_f16_init(self, config, module, *args, **kwargs):
    _orig_f16_init(self, config, module, *args, **kwargs)
    for submod in self.module.modules():
        if type(submod).__name__ == "Mamba3":
            for name in _MIMO_FP32_PARAMS:
                p = getattr(submod, name, None)
                if p is not None and p.dtype != torch.float32:
                    p.data = p.data.to(torch.float32)

Float16Module.__init__ = _cppmega_f16_init
```

Installed by `scripts/cppmega_fp8_shim.py` alongside the MIMO
`__post_init__` patch. This replaced the older per-forward hook after nsys
profiling showed that the repeated `.data.float()` copies were costing
~305 ms/iter. The one-shot restore is the live workaround that upstream
should understand when reading this pack.

## Suggested upstream fix

Two options, roughly ranked by disruption:

### (a) Per-parameter opt-out attribute (smallest change)

Teach `Float16Module.__init__` to honor a per-parameter
`_keep_fp32 = True` boolean attribute. Mamba3 (and any other module that
has a fp32-kernel-contract parameter, e.g. certain attention bias paths)
then tags those parameters at init time and Megatron leaves them alone:

```python
# In Megatron: Float16Module.__init__
for p in self.module.parameters():
    if getattr(p, "_keep_fp32", False):
        continue
    p.data = p.data.to(self.params_dtype)
```

This is non-invasive and composes with any other
fp32-contract-kernel module that adopts the tag.

### (b) Special-case known fp32 module types

Hardcode a list of module types whose parameters should not be cast,
initially `Mamba3`. Ugly but does not require an upstream change in
`mamba_ssm`.

We prefer (a) because it avoids Megatron needing to import / name-check
`mamba_ssm` types and lets third-party modules opt in locally.

## Note on the Mamba3 side of the fix

Mamba3 *could* also defensively re-cast in its own forward:
`DT = F.softplus((dd_dt + self.dt_bias).to(torch.float32))`. That's a
one-line change in `mamba_ssm/modules/mamba3.py` and the local convenience
patch `05_mamba3_dt_fp32_gqa_bwd.patch` includes it. But it only papers over
`DT`; `D`, `Q_BIAS`, `K_BIAS`, MIMO biases are all independently
cast by `Float16Module` and would each need their own defensive cast.
Fixing `Float16Module` once is cleaner.

## Evidence / shared reproducer note

We currently keep this bug's minimal evidence inside the shared pack-05
reproducer at `upstream_prs/examples/05_mamba3_dt_fp32_gqa_bwd/` because the
Megatron cast bug and the Mamba-side GQA bug co-trigger in the same lane.
For PR 16, the relevant stages are:

- `bf16` -> bug reproduced (`DT` rejected as bf16 by the TileLang kernel)
- `fp32` -> fix path validated (finite grads once fp32 contract is restored)

When filing upstream, do not present the shared local `.patch` as if it were
a single-repo fix bundle. Only the Float16Module contract part belongs in
Megatron-LM.

## Environment

- Megatron-core: 0.18.0rc0 (dev-latest), also seen on 0.15.0rc7
- PyTorch: 2.12 nightly + cu132
- Mamba-SSM: state-spaces/mamba fork with TileLang MIMO kernels
  (PyPI `mamba-ssm==2.3.1` does not ship `ops.tilelang` or
  `modules.mamba3`, so the bug surfaces only on the fork)
- NVIDIA H200 SXM (sm_90a), bench3 + europe
- Also reproducible on GB10 (sm_121f) for the forward-kernel symptom
  (backward path blocked there by an unrelated TileLang
  `LayoutInference` FloorMod crash — see pack 13)

## Impact on cppmega

**Critical.** Blocks NAM56R MIMO 7/7 training until the fp32-contract params
are restored after Megatron's blanket cast. Without the workaround, MIMO
path errors out on first forward with NaN grad norms from iteration 1.
