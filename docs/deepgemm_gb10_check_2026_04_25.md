# DeepGEMM / medmekk GB10 Check (2026-04-25)

## Result

`medmekk/deep-gemm` and upstream `deepseek-ai/DeepGEMM` are useful references,
but they are not drop-in GB10 (`sm_121`) kernels for cppmega today.

The implementation is not a Triton-only path that will naturally lower to GB10.
The current production DeepGEMM API is Python bindings over C++/CUDA JIT kernels
using CuTe/CUTLASS. The runtime can form an `sm_121a` target string, and local
CUDA 13.2 accepts `sm_121a` for an empty file, but the exported GEMM dispatch
checks exact architecture major `9` or `10`. GB10 reports major `12`, so calls
hit `Unsupported architecture` branches before a useful kernel is selected.

The upstream host extension did build/import on this GB10 stack, but a tiny
`fp8_gemm_nt` API smoke failed before useful GEMM dispatch with:

```text
RuntimeError: Assertion error (csrc/apis/layout.hpp:59): Unknown SF transformation
```

That matches the source: DeepGEMM scale-factor layout conversion has SM90 and
SM100 branches, but no SM12x branch.

The local diagnostic probe is kept at `tools/deepgemm_probe.py`. It is
non-invasive: it records Torch/CUDA/GPU state and scans DeepGEMM source trees
for architecture-dispatch evidence without importing `deep_gemm` or launching
GEMM kernels.

## Sources Checked

- Upstream `deepseek-ai/DeepGEMM` at `891d57b4db1071624b5c8fa0d1e51cb317fa709f`
  from 2026-04-24.
- Hugging Face `medmekk/deep-gemm` at
  `c67ae407a2e691113616e14f3284177aea6bcfd6` from 2026-02-16.
- Local DeepGEMM copies under `/home/dave/vllm/.deps/deepgemm-src`,
  `/home/dave/vllm/vllm/third_party/deep_gemm`, and FlashInfer wrappers.
- Prior nanochat notes in
  `/home/dave/source/nanochat/docs/fp8_deepgemm_notes.md`.

## Key Evidence

- `medmekk/deep-gemm` declares CUDA capabilities `9.0a` and `10.0a` only.
- Dense FP8/FP4, grouped FP8, K-grouped FP8 wgrad, BF16 GEMM, and scale-layout
  conversion all dispatch on exact `arch_major == 9` or `arch_major == 10`.
- SM100 kernels use Blackwell datacenter features such as TMEM/tcgen05. GB10's
  local hardware notes say those are not present on `sm_121a`.
- SM90 kernels use Hopper WGMMA/GMMA assumptions. Forcing that path on GB10 is
  not a sound compatibility plan.
- Vendored CUTLASS does include many SM120/SM121 headers and builders, so an
  SM120-family port is plausible. That is a port against CUTLASS/CuTe, not a
  one-line DeepGEMM enablement.

## nanochat Check

`../nanochat` does not appear to use the external `deep_gemm` package in active
code. It has research notes and local "DeepGEMM-style" work:

- `docs/fp8_deepgemm_notes.md` already marked SM121/GB10 as unsupported and
  identified MoE grouped GEMM plus K-grouped wgrad as the interesting APIs.
- `nanochat/fused_moe.py` has local Triton grouped-wgrad kernels and
  DeepGEMM-style contiguous payload construction. It borrows the layout idea;
  it is not evidence that external DeepGEMM ran successfully on GB10.

## Fit For cppmega

For dense MLP/projection GEMMs on GB10, keep probing TE `generic_gemm`,
MXFP8/NVFP4 quantizers, and SM120-family CUTLASS/CuTe/TileLang paths first.

For SparseMLA, DeepGEMM's dense/grouped GEMMs are the wrong API boundary. The
kernel has to consume sparse/top-k metadata, perform online softmax, apply
block scales inside K-block accumulation, and return the attention output plus
LSE. That points to a custom SparseMLA kernel, not a wrapper around
`fp8_gemm_nt`.

If an external DeepGEMM adapter is later added for H100/H200/B200, it should be
hard-gated to architecture major `9` or `10` and fail closed on `12.x`.
