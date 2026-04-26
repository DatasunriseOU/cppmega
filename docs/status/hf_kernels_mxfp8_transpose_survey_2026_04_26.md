# HF kernels MXFP8 transpose survey, 2026-04-26

Scope: candidates for GB10/SM120/SM121 block-scaled FP8/MXFP8/NVFP4 GEMM and the current TE MXFP8 backward problem. The specific target is not a general overview; it is a shortlist of kernels or source fragments we can realistically adapt for a TN-only Blackwell GeForce path, fused transpose/scale indexing, or no-materialization backward.

Local context: current GB10 notes show TE MXFP8 backward needs a TN adapter. Native TE `NN` dgrad and `NT` wgrad fail under cuBLASLt on GB10, while `TN` works. Pure Python view/stride retargeting of compact columnwise MXFP8 payloads produced wrong math; the missing piece is real reordered storage, or a lower-level kernel that changes compact scale indexing instead of materializing an intermediate.

## Search coverage

Used web/HF/GitHub search plus local clones under `/tmp/cppmega_kernel_research`. Brave and Tavily were unavailable during this pass due rate/credit limits; Perplexity long query timed out, so I used direct source inspection for conclusions.

Local copies:

| Source | Local path | Revision |
|---|---:|---:|
| NVIDIA CUTLASS | `/tmp/cppmega_kernel_research/cutlass` | `f74fea9ce358` |
| SGLang | `/tmp/cppmega_kernel_research/sglang` | `c7878dbb6ddf` |
| FlashInfer | `/tmp/cppmega_kernel_research/flashinfer` | `5e1318cb33eb` |
| b12x wheel | `/tmp/cppmega_kernel_research/b12x-0.7.0-wheel` | PyPI `0.7.0` |
| DeepGEMM upstream | `/tmp/cppmega_kernel_research/deepgemm` | `891d57b4db10` |
| HF kernels-community/deep-gemm | `/tmp/cppmega_kernel_research/hf-kernels-community-deep-gemm` | `a682ae320f71` |
| HF medmekk/deep-gemm | `/tmp/cppmega_kernel_research/hf-medmekk-deep-gemm` | `c67ae407a2e6` |
| TileLang | `/tmp/cppmega_kernel_research/tilelang` | `8f4a08f56de7` |
| lna-lab SM120 patch notes | `/tmp/cppmega_kernel_research/lna-blackwell-geforce-nvfp4-gemm` | `b2c224e0f156` |
| RightNow Tile | `/tmp/cppmega_kernel_research/rightnow-tile` | `88ed930e4943` |
| HF kernels-community/finegrained-fp8 | `/tmp/cppmega_kernel_research/hf-finegrained-fp8` | `9bc4bb7dba9f` |
| HF kernels-community/fp8-fbgemm | `/tmp/cppmega_kernel_research/hf-fp8-fbgemm` | `a8c3d2f157b9` |

Primary URLs:

- <https://github.com/NVIDIA/cutlass>
- <https://github.com/sgl-project/sglang>
- <https://github.com/flashinfer-ai/flashinfer>
- <https://pypi.org/project/b12x/>
- <https://github.com/deepseek-ai/DeepGEMM>
- <https://huggingface.co/kernels-community/deep-gemm>
- <https://huggingface.co/medmekk/deep-gemm>
- <https://huggingface.co/kernels-community/finegrained-fp8>
- <https://huggingface.co/kernels-community/fp8-fbgemm>
- <https://github.com/tile-ai/tilelang>
- <https://github.com/lna-lab/blackwell-geforce-nvfp4-gemm>
- <https://github.com/RightNow-AI/RightNow-Tile>

## Ranked shortlist

| Rank | Candidate | Fit | Effort | Risk | Perf potential | What to take |
|---:|---|---|---|---|---|---|
| 1 | CUTLASS SM120 block-scaled collectives/examples | Best upstream source for SM120 block-scaled TN GEMM and scale layouts | Medium-high | Medium | High | Use `Sm120BlockwiseScaleConfig`, `CollectiveBuilder`, and `79c/79d` argument/layout setup as the base for a cppmega PyTorch extension. |
| 2 | SGLang `sgl-kernel` dense SM120 FP8 blockwise op | Closest PyTorch-facing source-level wrapper for SM120 TN FP8 blockwise GEMM | Medium | Medium | High | Lift/adapt `fp8_blockwise_scaled_mm` wrapper and exact input/scale checks. Good first prototype for shape/layout plumbing. |
| 3 | FlashInfer `mm_mxfp8` / SM120 NVFP4 paths | Closest external black-box API for MXFP8/NVFP4 experiments | Low-medium for trial, medium-high for vendoring | Medium | High | Use API contract and scale-swizzle constraints; benchmark before source transplant. |
| 4 | b12x CuTe DSL SM120 NVFP4 + MXFP8 inline MMA | Best low-level SM120 CuTe DSL fragments for a custom fused scale-index kernel | High | High | High | Take SM120 warp-level design, scale layout view, and MXFP8 `mma.sync` inline asm helper. |
| 5 | DeepGEMM / HF deep-gemm | Strong API/layout reference, not SM120 | High | High on GB10 | High on SM90/SM100 only | Take layout utilities and k-grouped backward API ideas; do not port kernels directly to SM120. |
| 6 | TileLang SM100 MXFP8 example | Useful conceptually for scale-factor packing/transpose, not SM120 | High | High | Unknown on GB10 | Take packed UE8M0 quant/reference logic only; TCGEN05/TMEM kernel path is wrong ISA for SM120. |

No true drop-in was found for "TE compact columnwise MXFP8 payload + no materialized transpose + GB10 TN-only backward." The best realistic path is a CUTLASS/SGLang-derived SM120 TN kernel where we either:

1. feed pre-arranged operands/scales matching the SM120 scale layout, or
2. fuse the TE compact columnwise scale-index remap into the kernel's global/shared-memory load path.

## Candidate details

### 1. CUTLASS SM120 block-scaled GEMM

Local path: `/tmp/cppmega_kernel_research/cutlass`

Exact files:

- `examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu`
- `examples/79_blackwell_geforce_gemm/79d_blackwell_geforce_nvfp4_grouped_gemm.cu`
- `include/cutlass/gemm/collective/builders/sm120_blockscaled_mma_builder.inl`
- `include/cutlass/detail/blockwise_scale_layout.hpp`

Why it matters:

- `79c` is explicitly Blackwell SM120 block-scaled GEMM for GeForce RTX 50 and uses `mma.sync.aligned.block_scale`.
- `79c` sets `ArchTag = cutlass::arch::Sm120`, `OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp`, A row-major, B column-major, tile `128x128x128`, cluster `1x1x1`.
- `sm120_blockscaled_mma_builder.inl` has the critical static assert: only TN layout is supported. This exactly matches the GB10 constraint we are working around.
- The same builder derives `LayoutSFA/LayoutSFB` and SMEM scale layouts. That is the important code for changing compact scale indexing without Python view tricks.
- `79d` covers SM120 NVFP4 grouped GEMM and output scale-factor generation.

API/layout to copy:

- `layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1))`
- `layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1))`
- mainloop arguments `{A, stride_A, B, stride_B, SFA, layout_SFA, SFB, layout_SFB}`

Assessment: not a drop-in because examples are standalone and `79c` is mixed MXFP8/MXFP6 rather than our exact TE tensor wrapper. It is still the best source to build the real GB10 kernel.

### 2. SGLang `sgl-kernel` SM120 dense FP8 blockwise op

Local path: `/tmp/cppmega_kernel_research/sglang`

Exact files:

- `sgl-kernel/csrc/gemm/fp8_blockwise_gemm_kernel.cu`
- `sgl-kernel/python/sgl_kernel/gemm.py`
- `sgl-kernel/csrc/common_extension.cc`
- `sgl-kernel/tests/test_fp8_blockwise_gemm.py`
- `python/sglang/srt/layers/quantization/fp8_utils.py`

Public Python API:

```python
from sgl_kernel import fp8_blockwise_scaled_mm
out = fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype)
```

Observed constraints from the C++ wrapper:

- `mat_a`: 2D CUDA, row-major, `torch.float8_e4m3fn`
- `mat_b`: 2D CUDA, column-major, `torch.float8_e4m3fn`
- `mat_a.size(1) == mat_b.size(0)`
- `scales_a`: FP32, shape `(M, K / 128)`, M-major accepted via `stride(0) == 1`
- `scales_b`: FP32, shape `(K / 128, N / 128)`, K-major accepted via `stride(0) == 1`
- output dtype: FP16 or BF16

SM120 implementation uses CUTLASS `Sm120BlockwiseScaleConfig`, A row-major, B column-major, `KernelScheduleSm120Blockwise`, ping-pong fallback for small M, tile `128x128x128`, `ScalesPerTile = Shape<_128,_1,_1>`.

Why it matters:

- This is a much smaller PyTorch-extension-shaped source than CUTLASS examples.
- It gives exact runtime checks, padding behavior, and tests for varied M/N/K.
- It is FP8 blockwise with FP32 scales, not MXFP8 UE8M0 compact scales. So it is not a direct TE MXFP8 replacement, but it is a close prototype for TN-only backward mechanics.

Grouped caveat:

- `sgl-kernel/csrc/moe/fp8_blockwise_moe_kernel.cu` grouped path currently dispatches SM90/SM100 only in the inspected tree, not SM120.

### 3. FlashInfer MXFP8/NVFP4

Local path: `/tmp/cppmega_kernel_research/flashinfer`

Exact files:

- `flashinfer/gemm/gemm_base.py`
- `include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h`
- `flashinfer/jit/gemm/cutlass/generate_kernels.py`
- `include/flashinfer/gemm/mxfp8_gemm_cutlass_template.h`
- `flashinfer/gemm/kernels/dense_blockscaled_gemm_sm120_b12x.py`
- `flashinfer/quantization/kernels/nvfp4_quantize.py`

Relevant APIs:

```python
flashinfer.gemm.mm_mxfp8(a, b, a_descale, b_descale, out=None,
                         out_dtype=torch.bfloat16,
                         use_8x4_sf_layout=False,
                         backend="auto")
```

Important constraints:

- `mm_mxfp8` takes A `(m,k)` and B `(k,n)` column-major.
- For 2D non-swizzled scales, B descale must already be transposed as `(k // 32, n)`.
- On SM12x, CUTLASS MXFP8 backend only supports 1D swizzled scales with `SfLayout.layout_128x4`; linear 2D scales raise.
- NVFP4 SM120 paths include b12x and CUTLASS/cuDNN runner selection; b12x requires CUDA 13+, 128x4 scale layout, NVFP4 only.
- The SM120 grouped generator emits tile shapes `[128,128,128]`, `[128,128,256]`, `[256,128,128]`, `[128,256,128]`, with mixed FP8xFP4 restricted to `[128,128,128]`.

Assessment: good for black-box experiments and wrapper/API contracts. It still expects transposed/swizzled B scales, so it does not solve no-materialization backward by itself. For NVFP4, it is stronger than for MXFP8 because the SM120 FP4 templates are explicit.

### 4. b12x

Local path: `/tmp/cppmega_kernel_research/b12x-0.7.0-wheel`

Exact files:

- `b12x-0.7.0.dist-info/METADATA`
- `b12x/gemm/dense.py`
- `b12x/cute/fp4.py`
- `b12x/cute/utils.py`
- `b12x/quant/expert_fp4.py`
- `b12x/integration/tp_moe.py`

Why it matters:

- Package metadata says it is SM120-only CuTe DSL for NVFP4 dense GEMM, routed MoE, and paged attention.
- Dense GEMM kernel is built around SM120 warp-level MMA, no TMEM/tcgen05, no multi-cluster, cluster always `1x1x1`.
- `convert_sf_to_mma_layout` maps swizzled scale tensors to the 6D MMA-compatible layout `(32, 4, m_tiles, 4, k_tiles, num_groups)` as a strided view.
- `mxfp8_mma_m16n8k32_f32_e4m3` is a concrete SM120 MXFP8 inline-asm helper using:
  `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.f32.e4m3.e4m3.f32.ue8m0`

Assessment: not an MXFP8 dense GEMM drop-in, and it adds a CuTe DSL/CUDA 13/PyTorch 2.10 dependency. It is the best low-level source if we decide to write a custom fused scale-indexing kernel rather than wrap CUTLASS C++.

### 5. DeepGEMM and HF deep-gemm

Local paths:

- `/tmp/cppmega_kernel_research/deepgemm`
- `/tmp/cppmega_kernel_research/hf-kernels-community-deep-gemm`
- `/tmp/cppmega_kernel_research/hf-medmekk-deep-gemm`

Exact files:

- `README.md`
- `deep_gemm/__init__.py`
- `tests/test_fp8_fp4.py`
- `tests/test_layout.py`
- `deep_gemm/include/deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh`

Relevant APIs:

- `deep_gemm.fp8_gemm_{nt,nn,tn,tt}`
- `deep_gemm.fp8_fp4_gemm_{nt,nn,tn,tt}`
- `deep_gemm.m_grouped_fp8_gemm_{nt,nn}_contiguous`
- `deep_gemm.k_grouped_fp8_gemm_tn_contiguous`
- `deep_gemm.transform_sf_into_required_layout`
- `deep_gemm.get_mn_major_tma_aligned_packed_ue8m0_tensor`
- `deep_gemm.get_k_grouped_mn_major_tma_aligned_packed_ue8m0_tensor`

Why not rank higher:

- Upstream requirements list SM90 or SM100, not SM120.
- The SM100 implementation uses TCGEN05/TMEM/UMMA assumptions that SM120 does not have.
- README explicitly leaves input transposition/FP8 casting to the user and recommends fusing them into prior kernels independently.

What to take:

- API shape for `k_grouped_*_tn_contiguous`, because it is close to MoE weight-backward thinking.
- Packed UE8M0 and TMA-aligned scale layout utility semantics.
- Do not port SM100 kernels directly to GB10.

### 6. TileLang

Local path: `/tmp/cppmega_kernel_research/tilelang`

Exact files:

- `examples/gemm_sm100/gemm_mxfp8_blockscaled_1d1d.py`
- `tilelang/language/gemm_op.py`
- `tilelang/intrinsics/tcgen05_macro_generator.py`
- `tilelang/tileop/gemm/gemm_tcgen05.py`

Why it matters:

- It has an SM100 MXFP8 block-scaled example with packed UE8M0 scale factors and a Torch reference.
- The example includes a warp dedicated to scale-factor transpose via `T.tcgen05_sf_warp_transpose`.
- It demonstrates how to pack 4 E8M0 scale entries into one `uint32` and how to select `sf_id` per K stage.

Why not a direct candidate:

- The implementation is explicitly TCGEN05/TMEM for SM100. SM120 rejects that ISA path.
- There is an SM120 target detector, but I did not find an SM120 MXFP8/NVFP4 GEMM kernel in the inspected tree.

### 7. lna-lab Blackwell GeForce NVFP4 notes

Local path: `/tmp/cppmega_kernel_research/lna-blackwell-geforce-nvfp4-gemm`

Exact files:

- `README.md`
- `docs/sm120-architecture.md`
- `patches/flashinfer/01-sm120-grouped-gemm-tiles.md`
- `patches/vllm/03-cutlass-experts-fp4-sm120.md`
- `patches/quack/11-quack-gemm-sm120.md`

Why it matters:

- It is a practical map of SM120 constraints: warp-level `mma.sync`, TMA, native FP4/FP8 block scaling, no TMEM, no UMMA/tcgen05, 99 KB shared memory, cluster `1x1x1`.
- It names the same safe FlashInfer SM120 grouped GEMM tile restrictions found in current FlashInfer source.

Assessment: not kernel code to transplant, but useful validation that SM100 DeepGEMM/TileLang kernels are the wrong donor for GB10.

### 8. HF kernels-community/finegrained-fp8 and fp8-fbgemm

Local paths:

- `/tmp/cppmega_kernel_research/hf-finegrained-fp8`
- `/tmp/cppmega_kernel_research/hf-fp8-fbgemm`

Findings:

- `finegrained-fp8` exposes Triton W8A8 FP8 matmul/grouped/batched functions and fused activation quantization. It computes `C = A @ B.T` with block/tensor scales but is generic Triton, not SM120 MXFP8/NVFP4 block-scaled tensor-core code.
- `fp8-fbgemm` is a per-row FP8 quantizer copied from FBGEMM/Triton style, not a Blackwell block-scaled GEMM candidate.

Assessment: useful only as fallback/reference quantization code. Not worth adapting for the GB10 TN/MXFP8 issue.

### 9. RightNow Tile

Local path: `/tmp/cppmega_kernel_research/rightnow-tile`

Findings:

- It is a TypeScript/Next.js CUDA-to-cuTile transpiler and demo app.
- It has GEMM/FP8 pattern detection and generic codegen templates, but no concrete SM120 MXFP8/NVFP4 GEMM kernel to adapt.

Assessment: not actionable for this kernel task.

## Recommended next steps

1. Prototype with the SGLang dense op shape contract first, because it is already a PyTorch-facing SM120 CUTLASS wrapper:
   `fp8_blockwise_scaled_mm(mat_a, mat_b, scales_a, scales_b, out_dtype)`.
   This should answer whether the cppmega backward path can accept a TN-only SM120 kernel wrapper with our actual M/N/K and output dtype constraints.

2. In parallel, start the real source base from CUTLASS `79c` + `sm120_blockscaled_mma_builder.inl`, not from DeepGEMM/TileLang. The builder's TN-only static assert is the architecture-level constraint we need to embrace.

3. For no-materialization backward, focus on scale layout remap:
   compare TE compact columnwise scale layout against CUTLASS `Sm120BlockwiseScaleConfig::tile_atom_to_shape_SFA/SFB`.
   If physical storage is incompatible, add a fused load/index remap in the kernel or a small CUDA emitter similar in spirit to the current transpose prototype.

4. Use FlashInfer `mm_mxfp8` only as a black-box benchmark/API reference unless its 1D swizzled scale layout can be produced cheaply from our TE tensors.

5. Keep b12x as the fallback route for a custom CuTe DSL kernel if CUTLASS integration blocks on scale layout. The key pieces to take are `convert_sf_to_mma_layout` and `mxfp8_mma_m16n8k32_f32_e4m3`.

