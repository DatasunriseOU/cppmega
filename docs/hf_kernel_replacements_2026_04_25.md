# HF kernel replacement scan, 2026-04-25

Worker D worktree: `/home/dave/source/cppmega-hf-kernels-agent`.

This scan used primary sources only: Hugging Face Kernel Hub repos/cards and
GitHub source. I inspected local cppmega integration points for context but did
not run GPU training.

## Primary sources inspected

| Source | Revision inspected | Relevant upstream URL |
| --- | --- | --- |
| Hugging Face `kernels-community` source tree | `a8c39a2` | <https://github.com/huggingface/kernels-community> |
| `kernels-community/finegrained-fp8` | from `kernels-community@a8c39a2` | <https://huggingface.co/kernels-community/finegrained-fp8> |
| `kernels-community/fp8-fbgemm` | from `kernels-community@a8c39a2` | <https://huggingface.co/kernels-community/fp8-fbgemm> |
| `kernels-community/gpt-oss-triton-kernels` | from `kernels-community@a8c39a2` | <https://huggingface.co/kernels-community/gpt-oss-triton-kernels> |
| `kernels-community/flash-mla` | from `kernels-community@a8c39a2` | <https://huggingface.co/kernels-community/flash-mla> |
| `kernels-community/mamba-ssm` | from `kernels-community@a8c39a2` | <https://huggingface.co/kernels-community/mamba-ssm> |
| State Spaces `mamba` | `316ed60` | <https://github.com/state-spaces/mamba> |
| LinkedIn `Liger-Kernel` | `b8f093a` | <https://github.com/linkedin/Liger-Kernel> |
| Apple `ml-cross-entropy` | `b7a0279` | <https://github.com/apple/ml-cross-entropy> |

HF Kernel Hub loading is via `from kernels import get_kernel` and a repo id
such as `kernels-community/finegrained-fp8`.

## Executive result

There is no HF Kernel Hub drop-in for our hardest current blocker:
Transformer Engine MXFP8/NVFP4 `Linear` backward on GB10. The closest HF
candidate is `kernels-community/gpt-oss-triton-kernels`, but it is a forward
Triton MoE/GEMM package with MXFP helpers, not a TE-compatible autograd linear
replacement.

There is also no drop-in HF SparseMLA FP8/block-scaled training kernel. The
closest candidate, `kernels-community/flash-mla`, is relevant as sparse MLA
forward/decode source, but it is explicitly SM90a/SM100f, has no sparse
backward, and does not target GB10.

For Mamba, the most relevant source remains the direct `state-spaces/mamba`
repo that cppmega already wraps. The HF Hub `mamba-ssm` package is useful for
Mamba selective scan / Mamba2-style kernels, but it is not a better Mamba3
replacement.

For fused CE, the best GB10-compatible replacement remains Apple CCE, not an HF
kernel. HF/Liger is a usable mean/sum fused CE fallback, but current source
still has a reduction=`none` backward gap.

## Candidate matrix

| Problem | Candidate | Architecture support from source | API shape | Gaps | Recommendation |
| --- | --- | --- | --- | --- | --- |
| TE MXFP8/NVFP4 Linear backward on GB10 | `kernels-community/gpt-oss-triton-kernels` | CUDA/ROCm/XPU/CPU package. CUDA `has_native_mxfp()` is `compute capability >= 10`; GB10 cc12 will pass that check. | `matmul_ogs(x, w, bias, routing_data, gather_indx, scatter_indx, PrecisionConfig(...))`; MXFP through `downcast_to_mxfp` and `PrecisionConfig.weight_scale` / `act_scale`. | Forward only in tests (`test_bwd = False`); no autograd backward; not TE `QuantizedTensor`; not TE MXFP8/NVFP4 API or scale layout; no GB10-specific proof. | Use only as a small forward probe/reference for MXFP8/MXFP4 layout behavior on GB10. Do not replace TE training linear. |
| TE MXFP8/NVFP4 Linear backward on GB10 | `kernels-community/finegrained-fp8` | CUDA/ROCm/XPU package; no explicit SM guard in Python/Triton source. | `fp8_act_quant(x, block_size=128)`, `w8a8_block_fp8_matmul(A, B, As, Bs, [block_n, block_k], output_dtype)`. | FP8 E4M3 W8A8 forward matmul only; software block scales shaped like `(M,K/block)` and `(N/block,K/block)`; not MXFP8/NVFP4; no backward. | Not a TE replacement. Keep as a forward-only dense FP8 baseline if needed. |
| TE MXFP8/NVFP4 Linear backward on GB10 | `kernels-community/fp8-fbgemm` | CUDA/ROCm/XPU package. | `quantize_fp8_per_row(a, ...) -> (a_fp8, a_scale)`. | Quantization helper only; no GEMM, no backward. | Not a replacement. |
| SparseMLA FP8/block-scaled forward/backward | `kernels-community/flash-mla` | Build targets `9.0a` and `10.0f`; runtime checks say sparse forward is only SM90a/SM100f. No sm_120/sm_121 target. | `flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v=512, ...)` with `q [s_q,h_q,d_qk]`, `kv [s_kv,h_kv,d_qk]`, `indices [s_q,h_kv,topk]`; decode API `flash_mla_with_kvcache(...)` supports FP8 KV cache. | Sparse prefill is forward only; no sparse backward; no batch dim in prefill; BF16 q/kv for prefill; decode FP8 cache format is DeepSeek-specific, not cppmega block-scaled training; not GB10. | Use as H100/B200 sparse MLA forward reference only. Do not integrate for GB10 training. Keep cppmega TileLang FP8 fwd/bwd for now. |
| SparseMLA FP8/block-scaled forward/backward | `kernels-community/finegrained-fp8` | CUDA/ROCm/XPU package. | Dense W8A8 matmul APIs above. | No sparse attention/top-k indexing/softmax/backward. | Not relevant except for isolated FP8 matmul experiments. |
| Mamba kernels | `kernels-community/mamba-ssm` | CUDA package; build includes selective scan C++/CUDA kernels. | `selective_scan_fn(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False)`. | HF Hub source is mainly selective scan plus Mamba2 Triton helpers; not a Mamba3 drop-in. README warns the Hub package scope/dependencies are too broad. | Useful packaging source for Mamba selective scan/Mamba2, not for current Mamba3 issues. |
| Mamba kernels | State Spaces `mamba` direct source | No formal GB10 claim; Mamba3 SISO is Triton, MIMO is TileLang. Source performance notes are H100-oriented. | `mamba3_siso_combined(Q,K,V,ADT,DT,Trap,Q_bias,K_bias,Angles,...)`; `mamba3_mimo(Q,K,V,ADT,DT,Trap,Q_bias,K_bias,MIMO_V,MIMO_Z,MIMO_Out,Angles,...)`. | MIMO source warns about shared-memory over-allocation and TileLang compile failures when dimensions/hardware deviate; final state is non-differentiable side output. | Keep current cppmega direct wrapping. Pin/smoke-test the upstream commit rather than replacing through HF Hub. |
| Fused CE | `kernels-community/liger-kernels` / LinkedIn Liger | HF build lists CUDA/ROCm/XPU; Triton source has no GB10-specific guard. | `LigerFusedLinearCrossEntropyFunction.apply(input, weight, target, bias, ..., reduction="mean", ...)`. | HF source did not support reduction=`none`; current GitHub source returns per-token loss but still contains the explicit comment that reduction=`none` needs extra backward work, and backward scales gradients as if `grad_output` is scalar-like. | OK for scalar mean/sum fallback. Do not use as Megatron per-token loss drop-in. Keep cppmega workaround if Liger is selected. |
| Fused CE | Apple `ml-cross-entropy` CCE | Triton/CUDA implementation; no explicit GB10 guard found. | `linear_cross_entropy(e, c, targets, bias=None, ignore_index=-100, softcap=None, reduction="mean", shift=0, return_lse=False, ...)`. | Not an HF kernel. Needs adapter for Megatron tensor-parallel output layer semantics; source warns about Triton 3.2 gradient issue. | Best current GB10 fused CE replacement because source and tests cover reduction=`none` forward/backward. Keep as preferred fallback over Liger on GB10. |

## Details by problem

### 1. TE MXFP8/NVFP4 Linear backward on GB10

No inspected HF kernel is a drop-in for TE `Linear` training backward with
MXFP8 or NVFP4 tensors.

`gpt-oss-triton-kernels` is the closest technical candidate. It includes
MXFP quantization helpers:

```python
downcast_to_mxfp(src_tensor, out_quant_type, axis)
upcast_from_mxfp(tensor, scale, dtype, axis)
```

For MXFP4 it stores packed e2m1 values in `torch.uint8`; for MXFP8 it stores
`torch.float8_e4m3fn` or `torch.float8_e5m2`; scales are `torch.uint8` per
32-element group. Its GEMM entrypoint is `matmul_ogs`, with MXFP scales passed
through `PrecisionConfig.weight_scale` and `PrecisionConfig.act_scale`.

The important blockers:

- It is tested as forward only. The upstream test matrix sets
  `test_bwd = False`.
- Its API is not `torch.nn.Linear` or TE `Linear`; it expects its own routing,
  tensor wrapper, and scale wrappers.
- Its layout is not TE `Float8Tensor` / `Float8BlockQuantizer` /
  `MXFP8Quantizer` / `NVFP4Quantizer`.
- Its architecture predicate is generic `cuda_capability_geq(10, 0)`. That
  means GB10 cc12 is treated as native MXFP, but it does not answer the local
  GB10 `sm_120f`/`sm_121a` suffix issue documented in cppmega.

`finegrained-fp8` and `fp8-fbgemm` do not solve this. `finegrained-fp8` gives
E4M3 W8A8 forward matmul and activation quantization; `fp8-fbgemm` exports only
per-row FP8 quantization.

Recommendation: keep the TE/cuBLAS/CUTLASS path as the production training
route. If we add an HF probe, probe only `gpt-oss-triton-kernels` forward MXFP
cases on GB10 with small matrices and compare against BF16. Do not spend
integration time trying to route TE linear backward through these HF kernels.

### 2. SparseMLA FP8/block-scaled forward/backward

`flash-mla` is the only inspected sparse MLA candidate with a relevant
attention API. It implements DeepSeek sparse attention prefill and sparse
decode with FP8 KV cache. The sparse prefill API is:

```python
flash_mla_sparse_fwd(
    q,       # [s_q, h_q, d_qk], bf16
    kv,      # [s_kv, h_kv, d_qk], bf16
    indices, # [s_q, h_kv, topk], int32
    sm_scale,
    d_v=512,
)
```

The source enforces:

- SM90a or SM100f only; no GB10 target.
- `q` and `kv` dtype BF16 for sparse prefill.
- `d_qk == 512 or 576`, `d_v == 512`, `h_q == 64 or 128`.
- Sparse prefill is forward only.

This does not match cppmega `SparseMLA_FP8`, which needs training forward and
backward, accepts the local `[seq,batch,heads,dim]` path, saves FP8 tensors and
scales, and runs TileLang forward/backward kernels.

Recommendation: keep cppmega TileLang SparseMLA FP8 fwd/bwd for training. Use
`flash-mla` only as a primary-source reference for an H100/B200 inference or
prefill-only adapter. It is not a GB10 replacement and not a backward path.

### 3. Mamba kernels

The HF `mamba-ssm` kernel is primarily a packaging route for State Spaces
selective scan:

```python
selective_scan_fn(
    u, delta, A, B, C,
    D=None, z=None, delta_bias=None,
    delta_softplus=False,
    return_last_state=False,
)
```

It does include Mamba2-style Triton SSD helper modules in the Python tree, but
the compiled kernel build is selective scan. The HF README itself warns that the
Hub package scope is broad and probably should be narrowed.

For Mamba3, the exact relevant source is still direct `state-spaces/mamba`:

```python
mamba3_siso_combined(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias, Angles,
    D=None, Z=None, Input_States=None,
    chunk_size=64,
    return_final_states=False,
    cu_seqlens=None,
)

mamba3_mimo(
    Q, K, V, ADT, DT, Trap, Q_bias, K_bias,
    MIMO_V, MIMO_Z, MIMO_Out, Angles, D, Z,
    chunk_size, rotary_dim_divisor, dtype,
    return_state=False,
    cu_seqlens=None,
)
```

The MIMO source explicitly says it is optimized for H100 with
`seqlen=2048`, `nheads_qk=1`, `nheads=32`, `headdim_qk=128`,
`headdim_v=64`, `mimo_rank=4`, `chunk_size=16`, and warns about shared-memory
over-allocation / TileLang compile errors outside tested dimensions or
hardware.

Recommendation: keep cppmega's direct state-spaces integration for Mamba3. HF
Hub `mamba-ssm` is not a better replacement for current Mamba3 work. Treat
Mamba3 as a pinned upstream-kernel dependency with small smoke tests per target
architecture.

### 4. Fused CE

`liger-kernels` is the HF Kernel Hub candidate. It exposes fused linear CE and
has broad Triton backend declarations. It is not an exact Megatron replacement
for our use because per-token `reduction="none"` backward is not cleanly
supported. The current GitHub source has added a per-token forward result, but
the file still contains the explicit source comment that extra backward work is
needed for `reduction == "none"`, and its backward helper still scales stored
gradients after the fact.

Apple CCE is not an HF kernel, but it is the better adjacent replacement. Its
public API:

```python
linear_cross_entropy(
    e, c, targets,
    bias=None,
    ignore_index=-100,
    softcap=None,
    reduction="mean",
    shift=0,
    return_lse=False,
    ...
)
```

The CCE source handles `reduction == "none"` in forward and backward, and its
tests parameterize backward correctness over `["none", "mean", "sum"]`.

Recommendation: keep cppmega's current order for GB10 CE: native Megatron LCE
where supported, Apple CCE as the preferred GB10 fallback, and Liger only for
scalar mean/sum or through the existing mean-and-broadcast workaround.

## Final recommendations

1. Do not replace TE MXFP8/NVFP4 training linear backward with an HF kernel.
   The closest source, `gpt-oss-triton-kernels`, is forward-only and layout
   incompatible. Use it only for a small GB10 MXFP forward probe if needed.

2. Do not replace cppmega SparseMLA FP8 training with `flash-mla`.
   `flash-mla` is valuable primary source for sparse MLA forward/decode on
   SM90a/SM100f, but it has no sparse backward and no GB10 target.

3. Keep Mamba3 on direct `state-spaces/mamba`. HF `mamba-ssm` can package
   selective scan/Mamba2 kernels but does not supersede the current Mamba3
   integration.

4. Keep Apple CCE as the practical fused CE fallback on GB10. HF/Liger is
   acceptable only when a scalar reduction is enough or when cppmega applies
   its existing reduction workaround.

5. If we add a probe script later, the only high-value checks are small,
   no-training forward calls:
   `gpt-oss-triton-kernels.downcast_to_mxfp + matmul_ogs` on GB10 and
   `finegrained-fp8.w8a8_block_fp8_matmul` for dense FP8 reference behavior.
