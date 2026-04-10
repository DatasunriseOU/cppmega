# Upstream Bug Tracker

Fixes in upstream dependencies that affect (or may affect) cppmega.

---

## state-spaces/mamba PR #909 -- Cache ctx.saved_tensors in Mamba3 backward (FSDP activation checkpointing)

- **PR**: https://github.com/state-spaces/mamba/pull/909
- **Status**: Open (as of 2026-04-10)
- **File modified**: `mamba_ssm/ops/triton/mamba3/mamba3_siso_combined.py`
- **Severity**: Critical when using FSDP activation checkpointing with Mamba3 scan kernels

### Bug

`_Mamba3Function.backward()` calls `ctx.saved_tensors` **twice**: once to check
`len(ctx.saved_tensors)` and once to destructure the tensors.  When
`torch.utils.checkpoint` (activation checkpointing) is enabled, PyTorch registers
unpack hooks that only allow a **single** access to `ctx.saved_tensors`.  The second
call raises:

```
torch.utils.checkpoint.CheckpointError: torch.utils.checkpoint: Unpack is being
triggered for a tensor that was already unpacked once. If you are calling
ctx.saved_tensors in backward, make sure to do so only once. Otherwise please open
an issue with details on your use case.
```

This surfaces when running Mamba3 with FSDP + activation checkpointing (common in
large-scale Megatron/NeMo training).

### Fix (before/after)

**Before** (buggy -- two accesses to `ctx.saved_tensors`):

```python
# Line ~168 in _Mamba3Function.backward()
if len(ctx.saved_tensors) == 0:
    raise RuntimeError(
        "Backward called but forward ran without gradient tracking. "
        "Ensure inputs require grad or run under torch.enable_grad()."
    )

# ... later, line ~179:
(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
Final_SSM_State_save, cu_seqlens_save) = ctx.saved_tensors
```

**After** (fixed -- single access, cached in local variable):

```python
# Line ~168 in _Mamba3Function.backward()
_saved = ctx.saved_tensors                      # <-- cache once
if len(_saved) == 0:
    raise RuntimeError(
        "Backward called but forward ran without gradient tracking. "
        "Ensure inputs require grad or run under torch.enable_grad()."
    )

# ... later, line ~180:
(Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles, Angles_Cumsum,
D_save, Z_save, Input_SSM_State_save, Input_K_State_save, Input_V_State_save,
Out, Out_v, SSM_States, DA_CS, DA_CS_SUM, Q_rot, K_scaled, QK_dot, Scale, Gamma,
Final_SSM_State_save, cu_seqlens_save) = _saved  # <-- reuse cached
```

### Fix pattern

Cache `ctx.saved_tensors` in a local variable at the top of `backward()` and use
that local for all subsequent accesses.  This is the correct pattern for any
`torch.autograd.Function` that may run under activation checkpointing:

```python
@staticmethod
def backward(ctx, *grad_outputs):
    _saved = ctx.saved_tensors   # single unpack
    # use _saved everywhere, never ctx.saved_tensors again
```

### Impact on cppmega

**Low / no direct impact at present.** Analysis:

1. **CppMegaMamba3Mixer** (`cppmega/megatron/mamba3_mixer.py`) uses the **native
   Megatron SSD kernel** (`mamba_chunk_scan_combined`), not `_Mamba3Function`.
   It is unaffected.

2. **CppMegaMamba3TE** (`cppmega/megatron/mamba3_te_mixer.py`) calls
   `mamba3_siso_combined()` which internally dispatches through `_Mamba3Function`.
   This path **is affected** if activation checkpointing is enabled.  However,
   CppMegaMamba3TE is the deprecated "author kernels" path (see
   `docs/changelog.md`), and production training uses the native mixer path.

3. cppmega has **no custom `torch.autograd.Function` subclasses**, so there is no
   risk of the same bug pattern appearing in our own code.

**Action items**:

- When upgrading `mamba-ssm` pip dependency, ensure the installed version includes
  this fix (PR #909 merged, or commit `809b31ec`+).
- If CppMegaMamba3TE is ever re-enabled with FSDP activation checkpointing, this
  fix is mandatory.
