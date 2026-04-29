# Wave 3 Agent B MXFP8 GEMM-ready route notes

Worktree: `/home/dave/source/cppmega-wave3-agentB`
Branch: `wave3-agentB-mxfp8-route`
Base: `079f279 perf: speed grouped MXFP8 direct kernels`

## Prototype

Accepted prototype: cache FlashInfer/CUTLASS GEMM-swizzled rowwise MXFP8 scales on
the owning TE MXFP8 tensor in `cppmega.megatron.flashinfer_mxfp8_gemm`.

The cache key includes the compact scale tensor data pointer, storage offset,
shape, stride, device, logical rows/cols, and PyTorch tensor version.  It is
bypassed when TE already marks `_with_gemm_swizzled_scales=True`, so TE-emitted
swizzled scales still flow directly to the stock FlashInfer/CUTLASS path.

This is a producer/cache route, not another custom GEMM mainloop.  It removes
the repeated per-call compact-scale-to-layout_128x4 swizzle kernel for stable
rowwise operands while retaining the stock FlashInfer SM120 CUTLASS GEMM.

## Donor source conclusions

- FlashInfer SM120 MXFP8 accepts rowwise operands plus 1D GEMM-swizzled scale
  tensors (`SfLayout.layout_128x4`).  Its local SM120 source rejects 2D linear
  scales, so it is not a direct consumer of TE compact columnwise scales.
- CUTLASS Blackwell GeForce example 79c uses `mx_float8_t`,
  `OpClassBlockScaledTensorOp`, and `LayoutSFA/LayoutSFB` from
  `Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA/SFB`.  That confirms the fast
  stock route is the native swizzled-scale route.
- TE local source already carries `with_gemm_swizzled_scales`/`optimize_for_gemm`
  plumbing and grouped MXFP8 GEMM checks that require swizzled scales.  The
  remaining TE-side opportunity is to emit/cache the GEMM-ready scale layout at
  the producer instead of repairing it at GEMM call sites.
- `../nanochat` has useful GB10 and FP8 operational notes, but no drop-in SM121
  MXFP8 rowwise swizzled-scale GEMM donor.

Rejected donor paths this wave:

- HF DeepGEMM variants: SM90/SM100 focus, not a drop-in SM120 MXFP8 TE compact
  columnwise route.
- HF finegrained FP8/Triton rowwise kernels: not Blackwell MXFP8 block-scaled
  tensor-core kernels.
- ThunderKittens, CCCL, and cuLA: useful lower-level references, but no
  immediate TE compact-columnwise to stock SM120 MXFP8 GEMM route.

## 1024^3 timing

Command:

```bash
flock /tmp/cppmega_gpu_profile.lock -c "cd /home/dave/source/cppmega-wave3-agentB && PYTHONPATH=/home/dave/source/cppmega-wave3-agentB:/home/dave/flashinfer:/home/dave/TransformerEngine TORCH_EXTENSIONS_DIR=/tmp/cppmega_wave3_agentB_ext CPPMEGA_ALLOW_TE_MXFP8_SM12=1 python tools/probes/flashinfer_mxfp8_gemm_shape_probe.py --m 1024 --n 1024 --k 1024 --warmup 10 --iters 50 --try-all-tactics --include-bf16-ref --include-wrapper-cache-probe --include-scale-swizzle-timing"
```

Device: NVIDIA GB10, sm_121.

| Route | Median/mean ms | TFLOP/s | Notes |
| --- | ---: | ---: | --- |
| BF16 torch.mm | 0.03232 | 66.44 | reference timing |
| FlashInfer `mm_mxfp8`, pre-swizzled scales | 0.03785 | 56.74 | stock public API |
| FlashInfer direct runner tactic 0, pre-swizzled scales | 0.02009 | 106.90 | fastest 1024^3 stock route |
| Wrapper forced reswizzle every call | 0.04935 | 43.52 | two scale swizzle kernels per GEMM |
| Wrapper cached scales | 0.04104 | 52.33 | cache present on both operands |
| Compact rowwise scale swizzle, x | 0.00432 | - | one scale tensor |
| Compact rowwise scale swizzle, weight | 0.00430 | - | one scale tensor |

FlashInfer direct runner valid tactics:

- tactic 0: 0.02009 ms
- tactic 1: initialization failed with `Error Internal`
- tactic 2: initialization failed with `Error Internal`

The wrapper cache improves the default wrapper path by 16.8% versus forced
reswizzle for this small GEMM.  The direct runner remains the fastest stock
dispatch once scales are GEMM-ready.

## Real shape wgrad probe

Shape: `dy [16384, 3584]`, `x.T [3584, 16384]`, logical GEMM
`[3584, 3584, 16384]`.  Command batch wrote JSON under
`/tmp/cppmega_wave3_agentB`.

| Backend | Median ms | Mean ms | Extra bytes | Parity |
| --- | ---: | ---: | ---: | --- |
| `a_col_smem_b_tma_early` compact direct | 26.950 | 27.493 | 0 | exact |
| `te_emit_swizzled_stock` full sidecar | 7.538 | 7.572 | 62,390,272 | rel_l2 1.13e-5 |
| `streaming_swizzled_stock` stock GEMM + tiled producers | 4.066 | 4.191 | 18,350,080 | rel_l2 1.13e-5 |

`te_emit_swizzled_stock` setup time for this run was 2.048 ms before timed GEMMs.
The streaming stock route is 1.85x faster than full TE-emitted sidecar GEMM and
6.63x faster than the compact direct loader on the full wgrad shape, while
holding sidecar scratch to 18.35 MB instead of materializing 62.39 MB.

## Profiler counters

Profiler artifact:

- `/tmp/cppmega_wave3_agentB/nsys_flashinfer_cache_1024.nsys-rep`
- `/tmp/cppmega_wave3_agentB/nsys_flashinfer_cache_1024_cuda_gpu_kern_sum.csv`

Top CUDA kernel summary from the 1024^3 nsys capture:

| Kernel group | Instances | GPU time share |
| --- | ---: | ---: |
| FlashInfer/CUTLASS SM120 MXFP8 GEMM | 40 | 46.2% |
| BF16 CUTLASS GEMM | 10 | 18.4% |
| `swizzle_rowwise_scale_kernel` | 44 | 2.8% |

The 44 swizzle instances are expected from the probe construction: initial
pre-swizzle, forced-reswizzle wrapper timing, cached-mode cache population, and
explicit scale-swizzle timing.  Cached timed wrapper iterations reuse the cached
scale tensors after warmup.

## Verification

```bash
pytest -q tests/test_flashinfer_mxfp8_gemm.py
python -m py_compile cppmega/megatron/flashinfer_mxfp8_gemm.py tools/probes/flashinfer_mxfp8_gemm_shape_probe.py
```

Result: `33 passed`; py_compile passed.

## Next blocker

The next speed step is not another compact-direct loader tweak.  Stock
FlashInfer/CUTLASS wants rowwise payloads plus GEMM-swizzled scales, so the
producer needs to own that layout:

1. Validate TE producer/cache invalidation for any in-place scale buffer reuse;
   the prototype is safe for normal immutable quantizer outputs because it keys
   on tensor version, but a custom producer that mutates raw storage must
   invalidate explicitly.
2. Promote the direct FlashInfer runner tactic 0 route through the run profile
   only after full-model validation, because it is much faster than the public
   `mm_mxfp8` wrapper at 1024^3.
3. Finish TE-side native swizzled-scale emission for saved backward sidecars so
   dense Linear can avoid per-call transpose/scale repair rather than caching
   after the fact.

Relevant external docs/source:

- CUTLASS Blackwell functionality:
  https://docs.nvidia.com/cutlass/4.3.0/media/docs/cpp/blackwell_functionality.html
- CUTLASS example 79c:
  https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79c_blackwell_geforce_mixed_mxfp8_mxfp6_bf16_gemm.cu
- FlashInfer SM120 MXFP8 binding:
  https://github.com/flashinfer-ai/flashinfer/blob/main/flashinfer/data/csrc/mxfp8_gemm_cutlass_sm120.cu
