# Mamba3 / Mamba Kernel Web Research - 2026-04-29

Scope: docs-only research for H100/H200 Hopper paths kept separate from
B200/GB200 Blackwell paths. No production code changed.

## Executive readout

For cppmega NAM56R, the practical split is:

- Hopper H100/H200: keep the current Mamba2 SSD Triton scan path as baseline;
  test upstream `state-spaces/mamba` Mamba3 TileLang MIMO again only with a
  recent TileLang main/release that includes the TMA/no-WS and pipeline-before-
  LayoutInference fixes. Treat TMA + warp specialization as an opt-in candidate,
  not a default.
- Blackwell B200/GB200: keep separate SM100 paths. The strongest candidates are
  FlashInfer's SM100 `selective_state_update` and NVIDIA CUTLASS CuTeDSL
  Blackwell `mamba2_ssd` example. Do not try to force these into Hopper codegen.
- Hugging Face `kernels-community/mamba-ssm` is a useful packaging/probing
  source for Mamba/Mamba2 functions and prebuilt wheels, but it is not a
  Blackwell-specific answer by itself.

## Tooling notes

- Perplexity Deep Research timed out at 120s in this worker, matching the
  parent report. Perplexity Pro `gemini-3.1-pro` and Reasoning
  `gpt-5.2-thinking` were used for breadth, then facts were checked against
  primary GitHub/HF/arXiv/Together sources.
- Brave MCP returned a credit-exhaustion/rotation error in this environment, so
  primary web checks were done through Tavily, Exa, GitHub API, raw GitHub, HF,
  and web.open.

## TileLang upstream findings

Relevant upstream TileLang fixes/patterns now exist, but they change the
assumptions from the older local P1 notes.

| Item | Status on 2026-04-29 | Why it matters for Mamba3 kernels |
| --- | --- | --- |
| `#1458` floordiv/floormod Z3 prover fix | Merged 2025-12-17 | Reduces analyzer false proofs around signed floor div/mod. This is relevant to symbolic bounds and TMA stride/layout legality checks, but is not by itself a Mamba3 fix. |
| `#1533` LayoutInference replication/non-var complement | Merged 2025-12-25 | Closes `#1374`, where LayoutInference failed on a reduction pattern using `j // 32`. This is relevant to fragment/shared layouts with replication. |
| `#1840` TMA lowering without warp specialization | Merged 2026-02-25 | Decouples TMA from WS enough to avoid undeclared `mbarrier` when TMA is enabled while WS is disabled. This directly addresses the old "TMA requires WS" failure mode. |
| `#2002` pipeline-before-LayoutInference and tiled WS stabilization | Merged 2026-04-07 | Reorders TileLang lowering so producer/consumer WS and software pipeline rewriting happen before LayoutInference; hardens multi-version buffers, TMA stride validation, mixed TMA/cp.async barriers, and LayoutInference propagation. |

Current TileLang main patterns seen in primary code:

- `ProducerConsumerWarpSpecialized()` runs before `LayoutInference()` when
  `allow_warp_specialized()` is true.
- `PipelinePlanning()` and `InjectSoftwarePipeline()` also run before
  `LayoutInference()`.
- `LowerBlackwell2SM()` runs before `LayoutInference()` for Blackwell 2-CTA /
  TCGEN5MMA visibility.
- `LowerTileOp` records `tl.has_tma`; later optimization uses that instead of
  side-channel pass config.
- TMA eligibility is guarded by global stride checks: innermost stride must be
  contiguous, other global strides must be 16-byte aligned in bytes, and TMA
  box last dimension must be 128-bit aligned.
- Plain `T.copy` no longer auto-upgrades all global-to-shared loads to TMA in
  the newest main code; explicit `T.tma_copy` and pipeline planning are the
  safer pattern for new kernels.

Implication for cppmega: retest the previous Mamba3 TileLang TMA/WS candidate
against current TileLang before carrying local 3D-to-2D workarounds forward.
The old local failure, `Cannot detect TMA layout` for rank-3 shared descriptors,
may still require a shape/layout rewrite, but the pass order and TMA barrier
plumbing have materially changed upstream.

## Candidate matrix

| Candidate | Arch target | Kernel family | Fit for NAM56R | Testable path | Risk |
| --- | --- | --- | --- | --- | --- |
| Current `state-spaces/mamba` Mamba2 SSD Triton (`mamba_chunk_scan_combined`) | H100/H200, also runs elsewhere | `_chunk_cumsum`, `_chunk_state`, `_state_passing`, `_chunk_scan`, bwd variants | Baseline; local NAM56R no-conv status already routes M layers here and chunk-size 256 helped GB10 | Keep as control; profile post-autotune scan family on H200 | Stable, but does not use Hopper TMA/WS |
| HF `kernels-community/mamba-ssm` | Broad CUDA wheels, includes H100 filter on HF kernels page | `selective_scan_fn`, `mamba_inner_fn`, `selective_state_update`, `mamba_chunk_scan_combined`, `Mamba2` | Good packaging/probe source and fallback dependency source, not a custom perf answer | `pip install -U kernels`; `get_kernel("kernels-community/mamba-ssm")`; compare functions to installed `mamba_ssm` | No benchmarks on card; many prebuilt variants; B200 perf unknown |
| `state-spaces/mamba` Mamba3 TileLang MIMO | Hopper first; maybe Blackwell only after compiler validation | `mamba3_mimo_fwd.py`, `mamba3_mimo_bwd.py`, varlen variants; upstream currently disables TMA lower + WS in pass configs | Highest relevance to Mamba3 MIMO prefill. Previous local selective-fwd H200 was a wash, full bwd hit TileLang TMA layout failures | Re-run with current TileLang main, current Mamba main, H200 NAM56R shape, toggling TMA/WS and latest pass order | Compiler-sensitive; bwd smem and rank/layout constraints; should remain opt-in |
| `state-spaces/mamba` Mamba3 CuTe DSL step | Hopper/Blackwell CuTe-capable stacks | Decode/update step implementation using CuTe DSL and explicit shared/register layouts | Interesting for decode-like state update, less direct for current training scan bottleneck | Isolate step/update microbench; compare to Triton `selective_state_update` | CuTe DSL dependency surface; integration effort |
| FlashInfer `flashinfer.mamba.selective_state_update` | SM100/B200 path plus generic dispatch | Blackwell-optimized selective state update merged in PR #2387; repo also has `SSDCombined` and `ssd_kernel.py` | Strong Blackwell-only candidate. PR explicitly says SM100 kernel can compile on SM90 but performs poorly on Hopper | Add B200-only microbench for decode/update; dispatch by compute capability | Keep separate from Hopper; API/shape matching needed |
| NVIDIA CUTLASS CuTeDSL Blackwell `mamba2_ssd` example | SM100/B200 | Full Mamba2 SSD example with TCGEN05, TMEM, named barriers, Blackwell helpers, tile scheduler | Best source for Blackwell-specific SSD structure and scheduling ideas; likely not drop-in | Port/probe as standalone candidate first; compare against NAM56R Mamba2 SSD Triton on B200 | Example-level code, heavy CuTe DSL/CUTLASS 4.4+ dependency |

## Blackwell-specific paths to keep separate

Keep these behind an explicit SM100/B200 dispatch boundary:

- FlashInfer SM100 `selective_state_update`: PR #2387 says the new horizontal
  producer/consumer kernel moves the bottleneck away from warp-level reduction
  on Blackwell, and also says its SM90/Hopper performance is not good.
- CUTLASS CuTeDSL `examples/python/CuTeDSL/blackwell/mamba2_ssd`: uses
  Blackwell helpers, `tcgen05`, TMEM offsets, named barriers, explicit warp IDs,
  and optional 2-CTA instruction plumbing. These are not Hopper-equivalent
  implementation details.
- TileLang `LowerBlackwell2SM` / TCGEN5MMA paths. The pass order intentionally
  exposes Blackwell 2SM annotations before LayoutInference, so these should not
  be mixed into the H100/H200 candidate gate.

## Practical cppmega NAM56R plan

1. Baseline H200 with current Mamba2 SSD Triton path and delayed/post-autotune
   nsys filtering. The relevant kernels are the same local status file already
   identified: `_state_passing_*`, `_chunk_state_*`, `_chunk_scan_*`, and bwd dx.
2. H100/H200 candidate A: current TileLang main + current Mamba main +
   Mamba3 MIMO TileLang, no local production patch. First compile-only and
   microbench `mamba_mimo_fwd`, `mamba_mimo_bwd_fwd`, `mamba_mimo_bwd_bwd`.
   Then measure NAM56R shape. Only pursue if full fwd+bwd improves enough to
   matter; previous selective fwd-only H200 result was a wash.
3. H100/H200 candidate B: HF `kernels-community/mamba-ssm` as packaging/control
   source. Use it to check whether upstream wheels differ from the local
   installed `mamba_ssm` behavior; do not expect it to solve TMA/WS.
4. B200 candidate A: FlashInfer `selective_state_update`, decode/update
   microbench first. This is separate from training SSD scan, but likely useful
   for inference/state-update workloads.
5. B200 candidate B: CUTLASS CuTeDSL Blackwell Mamba2 SSD standalone probe.
   Compare against current Triton SSD at NAM56R-like `(B, L, H, P/G/N)` shapes.
6. Only after B200 candidates show clean shape/API parity should cppmega add an
   SM100-only dispatch. Hopper should keep its own dispatch and tuning.

## Sources

- Hugging Face kernels index: https://huggingface.co/kernels
- HF `kernels-community/mamba-ssm` card: https://huggingface.co/kernels-community/mamba-ssm
- HF kernels route for the same card: https://huggingface.co/kernels/kernels-community/mamba-ssm
- Together Mamba-3 blog: https://www.together.ai/blog/mamba-3
- Mamba-3 paper HTML: https://arxiv.org/html/2603.15569v1
- `state-spaces/mamba` Mamba3 TileLang directory: https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/tilelang/mamba3
- Mamba3 TileLang fwd raw: https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_fwd.py
- Mamba3 TileLang bwd raw: https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_bwd.py
- Mamba3 CuTe step: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/cute/mamba3/mamba3_step_fn.py
- Mamba2 SSD Triton combined: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_combined.py
- TileLang issue #1374: https://github.com/tile-ai/tilelang/issues/1374
- TileLang PR #1458: https://github.com/tile-ai/tilelang/pull/1458
- TileLang PR #1533: https://github.com/tile-ai/tilelang/pull/1533
- TileLang issue #1648: https://github.com/tile-ai/tilelang/issues/1648
- TileLang PR #1840: https://github.com/tile-ai/tilelang/pull/1840
- TileLang PR #2002: https://github.com/tile-ai/tilelang/pull/2002
- TileLang current phase pipeline: https://raw.githubusercontent.com/tile-ai/tilelang/main/tilelang/engine/phase.py
- TileLang current copy/TMA lowering: https://raw.githubusercontent.com/tile-ai/tilelang/main/src/op/copy.cc
- FlashInfer Blackwell selective-state-update PR: https://github.com/flashinfer-ai/flashinfer/pull/2387
- FlashInfer Mamba module: https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/mamba
- NVIDIA CUTLASS Blackwell Mamba2 SSD example: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/mamba2_ssd
