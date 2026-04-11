# NAM56R MTP optimization plan — 2026-04-11

**Status:** 6-agent MTP research completed 2026-04-11. Root causes identified for the 19% MTP overhead; optimization matrix designed to bring it to DeepSeek/Meta-reported 3-5% while keeping MTP architecturally enabled.

## The problem

Current VPP PP=2 VPP=2 MBS=4 GBS=64 baseline with MTP ON:
- **Iter time: 1963 ms**
- **Tok/sec: 112,152**
- **MTP overhead: 374 ms/iter (19.1%)** — measured by bench3 NoMTP removal experiment (+19% tok/sec)

**19% is 2-10× higher than published numbers.** Meta (arXiv 2404.19737) reports 0% with sequential detach pattern; DeepSeek-V3 (arXiv 2412.19437) reports ~2-5% for one MTP depth on a 61-layer model. Megatron-LM docs treat MTP as "one extra decoder layer of cost" ≈ 1/52 = 1.9% for our model.

## Root causes — 6 research agents converged on these findings

### 1. `--untie-embeddings-and-output-weights` in launch script (`remote_train_h200_nam56r_full.sh:122`)

Our launch script **UNTIES** embeddings from output head. DeepSeek-V3 **ties** them explicitly: *"for each MTP module, its embedding layer is shared with the main model… its output head is shared with the main model."*

With untied:
- MTP head gets its own `[hidden × vocab] = [3584 × 65536]` = **~235M extra parameters**
- Separate `[B×S, V]` GEMM per iter
- Extra gradient + optimizer state memory (~2-4× for Adam)
- No weight sharing benefit

**Fix: 1-line removal of `--untie-embeddings-and-output-weights`.** Megatron defaults to tied.

### 2. MTP layer is NOT in standalone VPP chunk

Megatron-LM supports `mtp_standalone` placement (layout pattern `"...tt|m|L"` per docs). When standalone, MTP gets its own virtual pipeline chunk → **interleaved 1F1B schedule overlaps MTP compute with main backward on different micro-batches.**

Current: MTP is appended to the last main chunk → chunk becomes fat (460ms + 374ms = 834ms) → pipeline bubble waits on this slow chunk for every micro-batch.

Standalone: each chunk ~460ms, MTP chunk ~374ms, they run in parallel across micro-batches via 1F1B interleaving. **Effective MTP wall-clock ≈ marginal bubble increase (~70ms) instead of full 374ms.**

**Expected savings: ~300 ms/iter → 1663 ms iter → ~157,700 tok/sec.**

### 3. No fused cross-entropy (Liger / Apple Cut-CE)

Megatron-LM's `process_mtp_loss()` does `output_layer(hidden_states) → compute_language_model_loss` which materializes the full `[B×S, V]` logits tensor. Liger-Kernel provides `fused_linear_cross_entropy` (Triton) that chunks the matmul and computes local log-sum-exp without ever storing the logits. This is **30-50% of the MTP cost** per Meta's paper.

Not a launch flag — needs integration work. Defer until flags #1 and #2 are applied.

### 4. `mtp_loss_scaling_factor` possibly using default 0.3

DeepSeek-V3 schedule: λ=0.3 for first 10T tokens, **λ=0.1 for final 4.8T tokens**. NeMo DeepSeek-V3 recipe default is `mtp_loss_scaling_factor=0.1`. Megatron-Core default is 0.1. Check our `build_nam56r_megatron_native_args` and launch scripts — if we're using a higher value, lower it.

Note: λ scales gradient magnitude, NOT FLOPs. Backward still runs in full. Smaller effect than #1 and #2 but free to adjust.

### 5. Is DeepSeek hiding MTP in DualPipe bubbles? **NO.**

User hypothesis falsified by Exa agent (`a6f1ea5ac59d459fb`). DualPipe source code (`dualpipe/dualpipe.py`, `dualpipev.py`) contains **zero MTP references** — grep confirmed. DualPipe overlaps (forward-of-microbatch-X with backward-of-microbatch-Y) and (MoE all-to-all with compute), NOT (main fwd with MTP fwd). DeepSeek pays MTP cost every step; they keep it cheap through (a) sharing embedding/output head, (b) depth=1, (c) λ schedule. No scheduling magic.

### 6. nanochat already has optimized MTP — ~0% overhead

Our sister project `nanochat` (`/Volumes/external/sources/nanochat/`) implements FastMTP-style shared-block MTP and measures **MTP3 +6% faster** than NoMTP on H200 NAM52 4.1B (1247 vs 1176 tok/s), with 3.4% better loss. Techniques used (`nanochat/mtp.py`):
1. Shared block — 1 `nn.Linear(2D→D)` + 1 Transformer Block **recursed K times** (not K separate blocks)
2. Weight-tied `wte` + `lm_head` passed into forward
3. Roll-and-mask static shapes (prevents K-way graph recompile)
4. Activation checkpointing inside the K-loop
5. Fused linear+CE per depth (Liger/CCE style)
6. Cadence scheduling (`mtp_cadence` dynamic skip; λ=0 → MTP forward skipped entirely)
7. mtp.* params → AdamW (not Muon)
8. bf16 `lm_head` with Megatron TP vocab-parallel marker preservation

## Optimization matrix — 8 variant sweep on 2× H200×8

**Baseline to beat: 112,152 tok/sec (VPP PP=2 VPP=2 MTP ON, current config)**

| # | Config | Untied→Tied | MTP standalone | MBS | MTP | Extra | Expected tok/sec |
|---|---|:-:|:-:|:-:|:-:|---|---|
| 1 | Current baseline (control) | untied | no | 4 | on | — | **112,152 (verify)** |
| 2 | Tied embeddings only | **tied** | no | 4 | on | — | ~125-135k |
| 3 | Standalone VPP only | untied | **yes** | 4 | on | — | ~140-155k |
| 4 | **Tied + standalone** | **tied** | **yes** | 4 | on | — | **~157-165k (primary target)** |
| 5 | Full fixes + MBS=5 | tied | yes | 5 | on | — | ~170-185k |
| 6 | Full fixes + MBS=6 | tied | yes | 6 | on | — | ~180-200k (if mem fits) |
| 7 | Full fixes + PP=4 VPP=2 | tied | yes | 4 | on | PP=4 | test bubble shift |
| 8 | NoMTP control (architectural regression) | tied | — | 4 | **off** | — | ~133k (sanity check) |

**Expected winner:** variant 5 or 6 (~170-200k). At 200k we're **1.25× from the 250k target** which remaining levers can close (CUDA graphs after Megatron bug fixes, TP=2).

## Variant selection logic

- Variants **2, 3** isolate the individual contribution of tied-embedding and standalone-VPP fixes
- Variant **4** is the primary product config
- Variants **5, 6** test MBS scaling on top — tied embeddings free ~235M params which frees activation memory
- Variant **7** tests whether PP=4 VPP=2 gives better bubble than PP=2 VPP=2 when MTP is properly placed
- Variant **8** is a control to verify we're measuring MTP cost correctly (should equal the earlier 133k NoMTP result)

## Hard constraints

- **MTP must stay ENABLED** in production variants (architectural feature, not optional knob). Variant 8 is control-only.
- **Real data** (`clang_semantic_4k_v10_train`), no `--mock-data`
- **Loss must converge** (iter 30 LM loss < 3.5, no NaN)
- **Correctness gate** on variant 2 (tied embeddings): if loss diverges, the existing checkpoint was trained with untied weights and we need fresh init. Expected: fresh init trains fine (DeepSeek uses tied from scratch).

## Deployment across 2× H200×8

**bench3 (LOCATION_1):** variants 1, 2, 5, 7
**europe (LOCATION_2):** variants 3, 4, 6, 8

Parallel execution across machines. Each variant = 30-iter training run (~5-10 min wallclock including JIT compile). Total sweep wallclock: ~60-80 min if well-parallelized.

## References

- Perplexity Pro gemini-3.1-pro research: DeepSeek-V3 architecture + DualPipe myth falsification (agent `a61122d36f9efd016`)
- Exa deep research: Meta + Medusa + FastMTP + theoretical cost (agent `a2e67295ebd8c3291`)
- Brave research: Megatron-LM + NeMo + Liger-Kernel (agent `ae61832d919145955`)
- Tavily NVIDIA docs: Megatron `multi_token_prediction.py` 1479 LOC, NeMo defaults (agent `ae27648f4fd77cdb3`)
- Exa DualPipe investigation: confirmed NO MTP scheduling in DualPipe source (agent `a6f1ea5ac59d459fb`)
- nanochat MTP patterns: `/Volumes/external/sources/nanochat/mtp.py` FastMTP shared-block design (agent `a81e6a5e3eb405192`)

## Papers

- [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) §2.2 MTP, §3.2.1 DualPipe
- [Meta Better & Faster Multi-Token Prediction](https://arxiv.org/abs/2404.19737) Gloeckle et al. ICML 2024
- [Medusa: Simple LLM Inference Acceleration](https://arxiv.org/abs/2401.10774)
- [FastMTP: Shared-block MTP with 2.03× speedup](https://arxiv.org/abs/2509.18362) Tencent, Sep 2025
- [MuToR: MTP needs registers](https://arxiv.org/abs/2505.10518) May 2025

## Upstream references

- [Megatron-LM multi_token_prediction.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py) (1479 LOC, production)
- [Megatron-LM MTP docs](https://github.com/NVIDIA/Megatron-LM/blob/main/docs/user-guide/features/multi_token_prediction.md)
- [NeMo DeepSeek-V3 recipe](https://docs.nvidia.com/nemo-framework/user-guide/25.11/llms/deepseek_v3.html)
- [Liger-Kernel fused_linear_cross_entropy](https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py)
- [DualPipe source](https://github.com/deepseek-ai/DualPipe) (confirmed no MTP awareness)

---

## MTP optimization results

### MTP Super flags experiment (europe, 3 variants)

| Variant | Config | Iter ms | Tok/sec | LM loss@30 | MTP loss@30 |
|---|---|---|---|---|---|
| V1 baseline | untied, MTP on | 2348.5 | 111,621 | 2.70 | 2.66 |
| V2 `mtp_use_repeated_layer=True` depth=1 | Super flags | 2355.8 | 111,275 | 2.81 | 2.54 |
| V3 `mtp_use_repeated_layer=True` depth=2 | Super flags | 2688.9 | 97,530 | 2.75 | 2.66/2.49 |

**Conclusion:** `mtp_use_repeated_layer=True` works for Mamba hybrid but gives 0% speedup at depth=1 (no-op when only 1 layer). At depth=2, -12.6% (expected -- shared weights save params not FLOPs). MTP overhead is forward+backward FLOPs, not param count.

### Tied embeddings experiment (europe)

| Config | Iter ms | Tok/sec | Delta |
|---|---|---|---|
| Untied (baseline) | 2348.5 | 111,621 | 0% |
| Tied | 2349.4 | 111,623 | -0.04% |

**Conclusion:** 0% effect on PP=2 hybrid. Megatron can't fuse embedding/output head across PP ranks.

### 8-variant MTP sweep results (europe + bench3)

| # | Name | Iter ms | Tok/sec | Status |
|---|---|---|---|---|
| 1 | Control untied | 2348.5 | 111,666 | baseline |
| 2 | Tied only | 2349.4 | 111,623 | 0% gain |
| 3 | Standalone VPP | -- | -- | BLOCKED (Megatron hybrid) |
| 4 | Tied + standalone | -- | -- | BLOCKED |
| 5 | Tied MBS=5 | -- | -- | OOM (~142/140 GB) |
| 7 | PP=4 VPP=1 | 3258.3 | 80,490 | -28% regression |
| 8 | NoMTP control | 1981.2 | 132,438 | +18.6% (confirms 133k) |

**Key findings from sweep:**
- Standalone VPP (variants 3, 4) BLOCKED: Megatron hybrid (`mamba_model.py:195-199`) explicitly does not support standalone MTP placement. PR #3377 confirms.
- NVIDIA Nemotron 3 Super sidesteps this by using PP=1 (no pipeline parallelism).
- MBS=5 (variant 5) causes OOM at ~142/140 GB -- no headroom even with tied embeddings.
- PP=4 VPP=1 (variant 7) is a -28% regression due to pipeline bubble increase.
- The original plan's primary target (variant 4, tied + standalone, ~157-165k) is unreachable without upstream Megatron changes to support standalone MTP on hybrid models.

### Liger fused CE for MTP (bench3)

| Metric | Standard CE | Liger fused | Delta |
|---|---|---|---|
| MTP time (4 depths fwd+bwd) | 178.8 ms | 483.2 ms | 2.7x SLOWER |
| Peak memory | 27.36 GB | 5.49 GB | -82% |

**Conclusion:** Liger saves 82% memory but 2.7x slower on H200 (poor tensor core utilization on chunked small-M GEMMs). DO NOT enable on H200. Valuable for memory-constrained hardware only.

### CUDA graphs experiments (bench3)

| Scope | Status | Tok/sec | Delta |
|---|---|---|---|
| Baseline (no graphs) | -- | 68,844 | -- |
| `--cuda-graph-scope attn` | PASS | 69,822 | +1.4% |
| `--cuda-graph-scope full_iteration` | FAIL | -- | MoE `.item()` blocker |
| `transformer_engine` | FAIL | -- | Same MoE blocker |

**Key discovery:** cppmega already has working per-module CUDA graph scope: `--cuda-graph-scope attn mamba moe_router moe_preprocess` (in `nam56r_nemo_recipe.py:287-296`). Full MoE graph needs `--moe-pad-expert-input-to-capacity`. **211k tok/sec production recipe used this exact config.**

### Megatron CUDA graph blocker patches (bench3)

1. MoE `.cpu()` at `token_dispatcher.py:295` -- keep on GPU (v2 preserves `.item()` for eager path)
2. `compute_dacs_segsum_triton` autotune -- fixed single config (8 autotune blocks collapsed)
3. `_broadcast_cu_seqlens` -- TP=1 bypass in `megatron/training/utils.py:566` (direct patch, not shim -- local function)
4. `tokens_per_expert.sum().item()` at line 306 -- additional D2H sync (identified, needs v3 fix OR per-module scope bypass)

### Nemotron 3 Super/Nano MTP research

- Super uses `MambaModel` + PP=1 + `mtp_use_repeated_layer=True` + `mtp_num_layers=2`
- Nano has NO MTP
- Standalone MTP explicitly NOT supported for hybrid model (`mamba_model.py:195-199`, PR #3377 confirms)
- NVIDIA sidesteps by using PP=1

### cuDNN LD_LIBRARY_PATH fix (bench3)

Root cause of ALL bench3 training failures in late session: system cuDNN 9.10.2 loaded before venv cuDNN 9.20.0. Fix: `export LD_LIBRARY_PATH=.../nvidia/cudnn/lib:$LD_LIBRARY_PATH` in `~/.bashrc`. Applied and verified on both machines.

### Revised optimization path

The original plan's primary target (variant 4: tied + standalone at ~157-165k) is **blocked** by Megatron's lack of standalone MTP support for hybrid models. Remaining viable levers:

| Lever | Expected effect | Status |
|---|---|---|
| Per-module CUDA graphs (existing recipe) | Already in 211k config | needs cuDNN fix applied |
| `--moe-pad-expert-input-to-capacity` | Enables full MoE CUDA graph | untested with fix |
| FP8 on MLA+MoE | +15-20% | pending |
| TP=2 PP=2 VPP=2 | +15-25% | medium effort |
| MTP removal (last resort) | +18.6% | architectural regression |
