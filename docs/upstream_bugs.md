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

---

## fast-hadamard-transform PyPI sdist is broken (1.0.4)

- **Package**: `fast-hadamard-transform==1.0.4` on PyPI
- **Status**: Broken as of 2026-04-11
- **Severity**: Blocker for DSA (DeepSeek Sparse Attention) which depends on it transitively via megatron-core
- **Upstream**: https://github.com/Dao-AILab/fast-hadamard-transform

### Bug

The PyPI source tarball for `fast-hadamard-transform==1.0.4` is missing
`csrc/fast_hadamard_transform.cpp`. Running `pip install fast-hadamard-transform`
triggers `setup.py build_ext` which fails with "no such file" for the C++ source.
GitHub source has all the files; only the packaged sdist is broken.

### Fix

Install directly from GitHub with build isolation disabled so the existing torch
install is used:

```bash
pip install --no-build-isolation 'git+https://github.com/Dao-AILab/fast-hadamard-transform.git'
```

### Impact on cppmega

**High** — DSA (DeepSeek Sparse Attention) is part of the NAM56R architecture
(A-ranks 0, 4, 8 of the 13 A-layers use DSA instead of MLA). Without
`fast_hadamard_transform`, DSA import fails and NAM56R MIMO 7/7 training cannot
start. Applied in the 56k tok/sec baseline run (see
`docs/nam56r_mimo7_baseline_2026_04_11.md`).

---

## TileLang nvrtc backend broken on CUDA 13.2 (CCCL conflict)

- **Package**: `tilelang` (whatever version cppmega pins)
- **Environment**: CUDA 13.2 with system CCCL 13.2.27 (bench3 `/mnt/data/venv`)
- **Severity**: Blocker for any TileLang kernel compiled via `TILELANG_EXECUTION_BACKEND=nvrtc`

### Bug

The bundled cutlass headers shipped inside the tilelang wheel include
`cute/container/array.hpp`, which conflicts with the system CCCL 13.2.27
tuple-interface headers via the `_LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR`
macros. nvrtc's include search order picks up both and the macro redefinition
fails compilation.

```
error: #error cuda::std redefinition ... _LIBCUDACXX_SPECIALIZE_TUPLE_INTERFACE_VECTOR
```

### Workaround

Use the NVCC-subprocess backend instead of nvrtc:

```bash
export TILELANG_EXECUTION_BACKEND=cython
```

**Note:** the docstring in `cppmega/megatron/mamba3_author_spec.py:13` currently
says *"For MIMO, set TILELANG_EXECUTION_BACKEND=nvrtc"*. **This is wrong on
bench3 cu13.2.** The successful 56k tok/sec MIMO 7/7 baseline run used
`cython` backend. The docstring should be updated.

### Impact on cppmega

**High** — blocks all MIMO training until either the backend env is set or
upstream tilelang strips the bundled cute headers / updates them to match
CCCL 13.2.27 tuple-interface macros.

### Upstream fix needed

File an issue against tilelang to either:
(a) stop vendoring `cute/container/array.hpp` when the system CCCL version is
newer, OR (b) update the bundled cutlass to a commit that matches the CCCL 13.2
tuple interface. The cython path works but loses the nvrtc JIT speed benefit.

---

## Megatron Float16Module casts Mamba3 fp32 bias/D/dt tensors to bf16

- **Package**: `megatron-core` (all versions through 0.18rc0)
- **File**: `megatron/core/transformer/module.py` (`Float16Module.__init__`)
- **Severity**: Silent correctness bug — causes NaN or dtype-mismatch crash in TileLang `mamba_mimo_fwd_kernel` under bf16 training

### Bug

`Float16Module.__init__` walks all parameters of the wrapped module and converts
them to bf16 (or fp16) indiscriminately. Upstream
`mamba_ssm.modules.mamba3.Mamba3` initializes these specific parameters in fp32
deliberately (the scan kernel contract requires them in fp32):

- `Q_BIAS`
- `K_BIAS` (which is `C_bias, B_bias`)
- `D`
- `dt_bias`
- `mimo_x_bias`, `mimo_z_bias`, `mimo_o_bias` (MIMO paths only)

Megatron's bf16 wrapper doesn't know about this Mamba3-specific fp32 contract
and silently casts them, breaking the TileLang `mamba_mimo_fwd_kernel` signature
which dispatches on dtype and produces NaN or dtype-mismatch on the first
forward call.

### Workaround

Install a forward pre-hook on every `Mamba3` module that re-upcasts the `.data`
of these parameters to fp32 each forward:

```python
_MIMO_FP32_PARAMS = (
    "dt_bias", "D",
    "B_bias", "C_bias",
    "mimo_x_bias", "mimo_z_bias", "mimo_o_bias",
)

def _mamba3_fp32_repair_hook(module, args, kwargs):
    for name in _MIMO_FP32_PARAMS:
        p = getattr(module, name, None)
        if p is not None and p.dtype != torch.float32:
            p.data = p.data.to(torch.float32)

for mod in model.modules():
    if type(mod).__name__ == "Mamba3":
        mod.register_forward_pre_hook(_mamba3_fp32_repair_hook, with_kwargs=True)
```

Installed by the cppmega shim `cppmega_fp8_shim.py` alongside the MIMO
`__post_init__` patch. Verified working in the 2026-04-11 56k tok/sec run.

### Clean fix (upstream, not implemented)

Teach `Float16Module` to honor a per-parameter `_keep_fp32 = True` attribute
annotation, or special-case known fp32-requiring layer types (Mamba3,
DeepNormRMS, certain attention bias paths). Both require an upstream Megatron PR.

### Impact on cppmega

**Critical** — blocks NAM56R MIMO 7/7 training until the pre-hook is registered.
Without this workaround the MIMO path errors out on first forward with NaN grad
norms from iteration 1.

---

## TransformerEngine 2.13 cuBLASLt FP8 `lda % 16` assertion on GB10 sm_121a

- **File**: `transformer_engine/common/gemm/cublaslt_gemm.cu:157`
  (`CanonicalizeGemmInput`)
- **TE version**: 2.13.0 (both GB10 and bench3 H200)
- **Status**: Hard crash on GB10 sm_121a; same stack **PASSES on H200** per
  `docs/fp8_path_status.md` (all 4 Mamba3 paths).
- **First observed**: 2026-04-11 during task #77 Stream C Blackwell feature
  sweep on NAM56R-half.
- **Severity**: Blocks FP8 hybrid training of NAM56R-half on GB10.

### Symptom

Running NAM56R-half (hidden=1792, ffn=9472, heads=14, noconv spec) with
`--fp8-format hybrid --fp8-amax-history-len 16 --fp8-amax-compute-algo max`
via `scripts/remote_train_gb10_nam56r_single.sh` (CUDA graphs off):

```
File ".../transformer_engine/pytorch/module/layernorm_linear.py", line 733, in backward
  gemm_out, *_, reduce_scatter_out = general_gemm(weight, ...)
File ".../transformer_engine/pytorch/cpp_extensions/gemm.py", line 197, in general_gemm
  out, bias_grad, gelu_input, extra_output = tex.generic_gemm(*args, **kwargs)
RuntimeError: /TransformerEngine/transformer_engine/common/gemm/cublaslt_gemm.cu:157
  in function CanonicalizeGemmInput:
  Assertion failed: ret.lda % 16 == 0.
  Leading dimension requirement on A for FP8 GEMM. Caller must pad.
```

Happens in the **backward pass** of a `TELayerNormColumnParallelLinear`. The
stack does not identify which layer instance; the error fires on the first
backward of iter 1.

### Why it's not a cppmega model-config issue

All NAM56R-half dims are 16-aligned:

| param | value | % 16 |
|---|---|---|
| hidden_size | 1792 | 0 |
| ffn_hidden_size | 9472 | 0 |
| num_attention_heads | 14 | — |
| q_lora_rank | 64 | 0 |
| kv_lora_rank | 64 | 0 |
| qk_head_dim | 64 | 0 |
| qk_pos_emb_head_dim | 32 | 0 |
| v_head_dim | 64 | 0 |
| moe_ffn_hidden_size | 896 | 0 |
| moe_shared_expert_intermediate | 1024 | 0 |
| num_experts | 16 | 0 |
| mamba_num_groups | 8 | — |
| mamba_state_dim | 128 | 0 |
| mamba_head_dim | 64 | 0 |
| padded_vocab_size | 65664 | 0 |

Reproduced with `mbs=2 gbs=2` AND `mbs=4 gbs=4` so not a token-count lda.

### Why it's GB10-specific

The identical `(spec, dims, TE 2.13.0, mamba_ssm 2.3.1, torch 2.12+cu132)`
stack runs **FP8 hybrid PASS** on bench3 H200 for all 4 Mamba3 paths (A, B,
C, D) per `docs/fp8_path_status.md`. Only the target device differs.

The crashing leading dimension comes from a weight slice that TE materializes
inside `CanonicalizeGemmInput` before the cublasLt call; cublasLt's FP8 kernel
on sm_90/cc 10.3 (Hopper/H200) accepts lda values that sm_121a's kernel
rejects. NVIDIA's cuBLAS 13.2 release notes explicitly call out MXFP8 and
NVFP4 tuning for DGX Spark (GB10), but **not** E4M3 hybrid — consistent with
the hybrid path being the rough one.

Adjacent evidence from `docs/gb10_software_stack.md`:
> FP8 cuDNN attention — Not optimized on cc 12.1 (cuDNN FP8 paths target cc
> 10.3 only)

### Workarounds considered (and rejected)

1. Change `mbs`/`gbs` — doesn't help, lda is weight-space not batch-space.
2. Switch `--fp8-format` to pure `e4m3` — same cublasLt path, same assertion.
3. Disable CUDA graphs — already off in the failing run (`--cuda-graph-impl
   none`).
4. Patch `cppmega_fp8_shim.py` — shim fixes Mamba3 fp32 bias/D casts, not TE
   internal lda alignment. Tested in the H200 fp8 matrix; doesn't help here.
5. Pad one of the hidden dims to a 32× multiple — would change model arch
   and violates the "no feature disable hacks" rule; also unclear *which*
   dim since all are already 16-aligned.

### Upstream next step

Open a Transformer Engine issue with a minimal repro: NAM56R-half dims +
`--fp8-format hybrid` on sm_121a. The assertion message "Caller must pad"
suggests the fix is in the CanonicalizeGemmInput path itself — TE should pad
internally when targeting sm_12x.

### Impact on cppmega

**Blocks FP8 training on GB10** for NAM56R-half (and by extension full
NAM56R). Does NOT block H200 or B200 — those are the production targets per
`docs/gb10_software_stack.md` ("Training on GB10 — only for smoke tests.
Production goes to H200 / B200"). Recommend: use GB10 for BF16 smoke only,
run FP8 on bench3 H200 per the existing FP8 path matrix.

