# cppmega

Megatron-first training framework for **NAM56R** — a 4.73B hybrid Mamba3 + MLA + DSA + MoE model.

---

## 1. Model: NAM56R — what we train

NAM56R is a **4.73B parameter hybrid Mamba3 + MLA + DSA + MoE** language model
designed for code and semantic understanding. It originated as a research fork
of Mamba3 crossed with DeepSeek-V3.2's sparse attention components, and has
since been extended with MTP heads, an ngram-hash embedding, and structure-aware
code signals. The training stack here is Megatron-first — no NeMo runtime, no
Lightning — but it consumes a NeMo-style recipe file for configuration.

### Architecture overview

- **52 layers** in a `*EME*EME*EMM*` repeating pattern:
  - `E` = Expert MoE layer
  - `M` = Mamba3 SSM layer
  - `A` = full attention (MLA) layer — 4 of these, anchoring long-range dependencies
  - `R` = ngram-hash embedding head (at the model entrance)
  - The 9 DSA layers (sparse attention with top-k selection) are interleaved
    with the MoE/Mamba blocks
- **Hidden dim 4096, 32 heads, head_dim 128, d_state 128, vocab 131072**
- **Sequence length 4096**, BF16 or FP8 tensorwise compute (per machine)

The 13 attention layers split as **9 DSA + 4 full MLA** — DSA is *not* optional,
it is the production configuration. The other 39 layers are Mamba3 SSM / MoE
(1 Mamba + 1 MoE alternation within each `*EME*` group, with occasional
`*EMM*` groups for a second Mamba hop).

### Why hybrid

- **Mamba3 SSM** gives O(N) sequence cost with constant state — the backbone
  of cheap long-context training.
- **Full attention (MLA)** layers anchor long-range token-to-token dependencies
  where SSM state compression is lossy.
- **DSA (DeepSeek Sparse Attention)** gives selective top-k attention — dense
  attention's expressive power on the tokens that matter, at a fraction of
  the compute.
- **MoE** (16 experts, top-k=4, +1 shared) scales parameters sub-linearly in
  compute: we get a 4.73B-param model for roughly the active-param cost of a
  much smaller dense one.

### Components

| Component | Detail |
|---|---|
| **Mamba3 MIMO** | rank=4, chunk=16, ngroups=8. TileLang fused fwd/bwd kernels |
| **MLA** (Multi-head Latent Attention) | 4 layers, q_lora=64, kv_lora=64, qk_pos=64. TileLang SparseMLA + FP8 dispatch via `sparse_mla_ops/` |
| **DSA** (DeepSeek Sparse Attention) | 9 layers, indexer top-k=256, lemyx fused FA+KL warmup kernel, IndexCache cross-layer reuse (3 Full + 6 Shared) |
| **MoE** | 16 experts top-k=4 + 1 shared expert, EP=4 or EP=8 distribution (per machine), grouped GEMM |
| **MTP** | 2 multi-token-prediction depths, via Liger fused linear CE |
| **ngram-hash embedding** | 2-gram and 3-gram hash bucketing; heads=8, table=500k, embed_dim=16 |
| **Structure-aware components** | code AST-derived signals feeding the embedding / mixer |
| **Float16Module wrapper** | with selective fp32 — `D` and `dt_bias` are kept as fp32 tensors inside the SSM for numerical stability (but parameter storage is bf16; `.float()` happens in the forward). See "No _apply guard" below. |

---

## 2. Features included — always-on divergence from upstream

Every item below is a patch we apply on top of Megatron-LM / Liger / mamba_ssm
*unconditionally*. There are no silent fallbacks: if any patch fails to apply,
training crashes. The list is short on purpose; opt-in flags live in the
"Env-Gated" section further down.

### Mamba LinearCE class-swap — `apply_linear_ce_patch.py`
- **What**: replaces `MambaModel.output_layer` with `LinearCrossEntropyModule`
  so the main LM head goes through fused linear-CE instead of materializing
  the `[s*b, V=131072]` fp32 logits tensor.
- **Bug fixed**: upstream Megatron PR #3226 was silently reverted by PR #3207,
  re-breaking the fused path for MambaModel. Without this patch, MBS=12 runs
  OOM on ~12 GiB of fp32 logits.
- **Why ship**: unconditional memory win, zero throughput cost. Always on.

### MTP Liger fused CE — `mtp_liger_ce.py`
- **What**: replaces the vanilla MTP prediction head's `[s*b, V]` logits
  materialization with Liger fused linear CE, using `reduction="mean"` plus
  a broadcast trick to reconstruct per-token loss.
- **Bug fixed**: Liger issue #968 — FLCE `reduction="none"` silently corrupts
  the backward gradient. The `reduction="mean"` workaround yields correct
  gradients.
- **Why ship**: saves ~21 GiB of MTP-head activations at MBS=10 NAM56R.
  Always on.

### DSA indexer fused — `dsa_indexer_fused_patch.py`
- **What**: per-head `bmm` accumulation inside `_compute_index_scores`,
  eliminating the `[sq, b, h, sk]` fp32 intermediate tensor.
- **Bug fixed**: stock Megatron einsum materializes the full 4D scores tensor
  for every DSA layer; across 9 DSA layers that is ~45 GiB resident at MBS=10.
- **Why ship**: saves ~40 GiB at MBS=10. Verified correctness on GB10 to
  `rel_err = 1.6e-7`. Always on.

### DSA CG-safety patches — `apply_dsa_cg_patches.py`
- **What**: 10 patches (1-9 + 9b) enabling CUDA Graph capture in the DSA path.
  Replaces CPU-sync `torch.equal` / `torch.any` validation with branchless
  variants; relaxes hard-coded 576/512 dimension asserts to our 128/64;
  routes FP8 quantized tensors to `SparseMLA_FP8` zero-copy via `_data`;
  removes a stray `query.dequantize()` / `key.dequantize()` that had drifted
  into the installed `dsa.py`.
- **Why ship**: mandatory after every Megatron rebase. Without these the
  training either crashes or silently falls off the CUDA-graph path.

### Regional torch.compile — `mamba3_compile_patch.py`
- **What**: compiles 4 Mamba3 elementwise regions (the data-dependent A
  computation and neighbours).
- **Why ship**: 5.93× speedup on the data-dep-A region, ~6% end-to-end
  throughput. Always on.

### GB10 smem preflight — `preflight_smem_check.py`
- **What**: on `sm_121` machines (GB10), refuses to launch training unless
  *every* TileLang kernel in the import set was built with
  `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True` (keeping dynamic smem under
  the 99 KiB GB10 cap).
- **Why ship**: without this, kernels silently compile but fail at first
  launch with a shared-memory-too-large error. Hard-fail is better than
  guessing which kernel blew the limit.

### Index cache + lemyx DSA — opt-in

Not always-on, but worth listing here because they are the canonical DSA
training acceleration path:

- `CPPMEGA_INDEX_CACHE=1` — reuses indexer top-k across DSA-Shared layers
  (3 Full + 6 Shared pattern), 67% indexer compute savings.
- `CPPMEGA_LEMYX_DSA=1` — loads the lemyx/tilelang-dsa fused FA +
  lightning-indexer + KL kernel for the indexer warmup phase (~first 1000
  steps).

---

## 3. What was removed and why

The history is also the truth. These are paths we tried, measured, and
deleted — or kept in tree but gated off pending further work. Be honest about
each one.

### Deleted

- **`dsa_fp8_indexer.py`** *(deleted 2026-04-13)* — FP8 quantization for DSA
  indexer scores. Empirically **-0.91× on GB10**, **~1.07× projected on
  H200** — net regression. The expected memory savings did not materialize
  because the indexer is not memory-bound; compute quant overhead dominated.

- **`dsa_fp8_patch.py`** *(deleted 2026-04-13)* — umbrella FP8 dispatcher
  for the DSA path. Superseded by `apply_dsa_cg_patches.py` Patch 9
  (quantized tensor zero-copy into SparseMLA_FP8) + `dsa_indexer_fused_patch.py`
  (BF16 fused accumulation). No single FP8 file owns the DSA path anymore.

- **`dsa_tilelang_fused_kl.py`** *(deleted)* — standalone fused FA+KL
  TileLang kernel. Replaced by `lemyx_dsa_warmup.py`, which loads the same
  kernel via the upstream lemyx library's interface rather than a local copy.

### Kept but gated OFF by default

- **`mtp_native_hopper_ce.py`** — a native Hopper CuTe-DSL linear-CE kernel
  for the MTP heads. Infrastructure is committed; activation
  (`CPPMEGA_MTP_NATIVE_HOPPER_CE=1`) produces `grad_norm=NaN`. Suspects #1
  (transpose round-trip) and #2 (CG collective corruption) were empirically
  refuted. Suspects #3 (shared-weight dual-bwd), #4 (mask handling), and
  #5 (dtype mismatch) are under investigation. Will ship once root-caused.

- **DualPipeV V-shape pipeline** (`dualpipev_schedule.py`) — experimental
  pipeline schedule that carves 2 PP ranks inside a PP=1 Megatron config.
  **Incompatible with EP>1**: the DeepEP all-to-all desyncs across the
  V-shape rank groups. Gate `CPPMEGA_DUALPIPEV=1`, default off. Upstream
  NVIDIA PR #1524 (DualPipeV integration into Megatron) has been abandoned.

- **combined_1f1b overlap** (`hybrid_schedule_plan.py`) — closed as a
  production path: OOM at every PP=2 MBS we tried; PP=4 impossible with
  52 layers (pipeline bubble too deep). Code kept for reference, gate
  `CPPMEGA_EP_OVERLAP=1`, default off. EP within a PP stage on NVLink
  does not need compute/comm overlap; the NVIDIA ModelZoo DS-V3 recipe
  confirms this (Qwen3-30B-A3B uses PP=1 EP=8 with no overlap).

---

## 4. Production configurations + how to launch

Canonical production numbers and launch commands live in
**[docs/production_status.md](docs/production_status.md)** — single source of
truth for per-machine env vars, launch commands, and peak memory. The table
and examples below are a quick summary; any conflict with `production_status.md`
is a bug in this README.

Session 3 summary (2026-04-14, post Liger grad-corruption audit):

| Machine | Precision     | Topology             | Throughput       |
| ------- | ------------- | -------------------- | ---------------- |
| europe  | BF16          | PP=1 EP=4 MBS=8, CG off | **289 TFLOP/s** (gold record) |
| bench3  | FP8 tensorwise | PP=1 EP=8 MBS=10 v3, CG off | **268 TFLOP/s** (MTP Liger CE) |

Both machines use Liger `reduction="mean"` broadcast workaround for correct
gradients (Liger #968). The earlier "269.4 TFLOP/s bench3" measurement with
`reduction="none"` is **superseded** — it was silently corrupted.

Note: the topologies above are the *measured best* per machine — they differ
(europe=BF16 EP=4 MBS=8, bench3=FP8 EP=8 MBS=10). The default launcher
`scripts/remote_smoke_h200_dsa_9_4_m.sh` ships with the older "Stream L"
defaults (PP=2 VPP=2 MBS=4 GBS=64, VARIANT=v1 → EP=4); production topologies
are reached via env-var overrides — see Quick Start below and
`docs/production_status.md` for the exact launch command per machine. The
script is designed to run as the remote body inside a tmux session on either
bench3 or europe — there is no gcloud/scp wrapper in it.

| Parameter    | Value                                                     |
| ------------ | --------------------------------------------------------- |
| Model        | NAM56R 4.73B (52 layers, hybrid `*EME*EME*EMM*`)          |
| Architecture | heads=32, hidden=4096, headdim=128, d_state=128           |
| Mamba        | Mamba-3 MIMO rank=4, chunk=16, SISO+MIMO TileLang kernels |
| Attention    | 4 MLA layers (q_lora=64, kv_lora=64, qk_pos=64)           |
| DSA          | 9 DSA layers (sparse attention, indexer topk=256)         |
| MoE          | 16 experts, topk=4, shared expert, grouped GEMM           |
| MTP          | 2 prediction layers                                       |
| Precision    | BF16 (europe) / FP8 tensorwise (bench3) — see production_status.md |
| Parallelism  | PP=1, TP=1, EP=4/8 per machine (CG must be OFF at PP=1)   |
| Micro-batch  | MBS=8/10 per machine, seq_len=4096                        |
| Hardware     | 8x NVIDIA H200 141GB (NVLink)                             |

### Quick Start — prerequisites

```bash
# Megatron-LM: our local `dev_latest` branch on europe (based on 0.16.0rc0),
# which cherry-picks the currently-open upstream PRs #3674 and #4268.
# Neither PR is merged upstream yet — this branch is our local state.
# bench3 instead runs 0.18.0rc0 installed via pip-git @ 980211ae (upstream dev
# HEAD 2026-04-09). See Megatron Version section below for details.
cd /mnt/data/cppmega-root/megatron-lm   # bench3 path
# cd /home/dave/cppmega-root/megatron-lm  # europe path

# After any Megatron update, MUST apply patches:
cd /mnt/data/cppmega-root/cppmega
python -m cppmega.megatron.upstream_patches.apply_dsa_cg_patches

# Required packages
pip install dualpipe  # from github: pip install git+https://github.com/deepseek-ai/DualPipe.git
pip install liger-kernel
pip install apex  # NVIDIA apex from source with --cpp_ext --cuda_ext
```

### Quick Start — training

Paths below assume **bench3** (`/mnt/data/cppmega-root`). On **europe** the
root is `/home/dave/cppmega-root`; on either machine, run from inside the
appropriate tmux session (`bash -l`).

```bash
# bench3
cd /mnt/data/cppmega-root/cppmega
# europe
cd /home/dave/cppmega-root/cppmega
```

**Script default topology** (what you get with no overrides):
`TP=1 PP=2 VPP=2 MBS=4 GBS=64 MTP=2`, `VARIANT=v1 → EP=4`. Measured ~193
TFLOP/s on europe. This is the Stream L baseline the script was originally
written for.

```bash
# Script default (PP=2 VPP=2 MBS=4 EP=4) — Stream L topology, ~193 TFLOP/s europe
# NOTE: MTP Liger CE + Mamba LinearCE class-swap + DSA indexer fused are now
# unconditional (commit dd4da34) — no env gate needed for those.
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_LEMYX_DSA=1 \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh
```

**Production topology** — production configs differ per machine; see
`docs/production_status.md` for the canonical launch commands. Summary:
europe = BF16 PP=1 EP=4 MBS=8 (289 TFLOP/s); bench3 = FP8 tensorwise PP=1
EP=8 MBS=10 v3 (268 TFLOP/s). The launcher script honours env-var overrides:

```bash
# europe (289 TFLOP/s, BF16, PP=1 EP=4 MBS=8)
# NOTE: MTP Liger CE, Mamba LinearCE class-swap and DSA indexer fused are
# unconditional since dd4da34 — no env gate needed.
PP_SIZE=1 VPP_SIZE=1 MBS=8 EP_SIZE_OVERRIDE=4 \
CG_FLAGS=NONE \
CPPMEGA_INDEX_CACHE=1 \
CPPMEGA_LEMYX_DSA=1 \
CPPMEGA_LINEAR_CE_KERNEL=liger \
EXTRA_FLAGS="--cross-entropy-loss-fusion --cross-entropy-fusion-impl linear" \
bash scripts/remote_smoke_h200_dsa_9_4_m.sh
```

CG must be off at PP=1 (TE CG private pool = 39.5 GiB, which OOMs the
production topology). The script does *not* auto-disable CG when PP_SIZE=1;
you must pass `CG_FLAGS=NONE` explicitly as shown above.

The script auto-applies (regardless of topology):
- Regional torch.compile (4 Mamba3 elementwise regions)
- MTP Liger chunked CE (always on, no gate; saves 21 GiB MTP logits)
- Mamba LinearCE class-swap (always on, fixes upstream PR #3226→#3207 regression)
- DSA indexer fused per-head bmm (always on, saves ~40 GiB at MBS=10 NAM56R)
- IndexCache (67% DSA indexer savings) — when `CPPMEGA_INDEX_CACHE=1`
- lemyx fused FA+KL kernel (DSA warmup) — when `CPPMEGA_LEMYX_DSA=1`
- FP8 tensorwise recipe — when `FP8_FLAGS=...` is set (BF16 wins on europe)
- expandable_segments (mandatory, hard-coded)

## 5. Data preparation

See **[docs/data_preparation.md](docs/data_preparation.md)** for tokenizer
choice, corpus layout, ngram-hash bucket construction, and the Megatron
indexed-dataset build steps. (Stub; parallel agent is authoring this doc.)

## 6. Optimization stack — full patch reference

### Always On (no env gates) — patches install unconditionally; raise on failure

| Component              | File                                | Effect                                                                 |
| ---------------------- | ----------------------------------- | ---------------------------------------------------------------------- |
| Regional torch.compile | `mamba3_compile_patch.py`           | Fuses Mamba3 elementwise ops (5.93x data-dep-A)                        |
| expandable_segments    | script env var                      | Prevents CUDA allocator fragmentation OOM                              |
| Unfused DSA banned     | `apply_dsa_cg_patches.py`           | Crash if fused SparseMLA unavailable                                   |
| Mamba LinearCE swap    | `apply_linear_ce_patch.py`          | MambaModel.output_layer → LinearCrossEntropyModule (PR #3226→#3207 fix)|
| MTP Liger CE           | `mtp_liger_ce.py`                   | Chunked fused CE, saves 21 GiB MTP logits, reduction="mean" + broadcast |
| DSA indexer fused      | `dsa_indexer_fused_patch.py`        | Per-head bmm, saves ~40 GiB at MBS=10 NAM56R                           |

### Env-Gated (selectors + opt-in features)

| Component         | Gate                          | File                         | Effect                                               |
| ----------------- | ----------------------------- | ---------------------------- | ---------------------------------------------------- |
| IndexCache        | `CPPMEGA_INDEX_CACHE=1`       | `index_cache_patch.py`       | 3 Full + 6 Shared DSA layers = 67% indexer savings   |
| lemyx DSA         | `CPPMEGA_LEMYX_DSA=1`         | `lemyx_dsa_warmup.py`        | Fused FA+KL TileLang kernel for indexer warmup       |
| Linear CE kernel  | `CPPMEGA_LINEAR_CE_KERNEL=auto\|liger\|cce` | `apply_linear_ce_patch.py` | Selector for non-Blackwell fallback kernel; default `auto` |
| Selective FP8 MoE | `CPPMEGA_SELECTIVE_FP8_MOE=1` | `selective_fp8_moe_patch.py` | FP8 only on MoE expert GEMMs                         |
| Mamba recompute   | `CPPMEGA_MAMBA_RECOMPUTE=1`   | `mamba_recompute_patch.py`   | Activation checkpointing for Mamba layers            |
| FP8 param-gather  | `CPPMEGA_FP8_PARAM_GATHER=1`  | Megatron `--fp8-param-gather`| -5 GiB (FP8 all-gather bucket, master stays FP32)    |
| DualPipeV         | `CPPMEGA_DUALPIPEV=1`         | `apply_dualpipev_patch.py`   | V-shape PP; forces Megatron PP=1 (experimental)      |
| MTP native Hopper | `CPPMEGA_MTP_NATIVE_HOPPER_CE=1` | `mtp_native_hopper_ce.py` | **DO NOT ENABLE** — produces grad_norm=NaN, Suspects #1+#2 refuted, #3-5 pending |
| Mamba3 MIMO P1    | `CPPMEGA_MAMBA3_P1=1`         | `apply_mamba3_mimo_p1_patches.py` | TMA + warp-spec on Mamba3 MIMO fwd only (bwd blocked by TileLang TMA layout bug) |
| Prefer native Hopper | `CPPMEGA_PREFER_NATIVE_HOPPER_CE=1` | `apply_linear_ce_patch.py` | When local Megatron PR #3345 is cherry-picked, skip Liger/CCE reroute and use native Hopper CuTe DSL kernel |

### Pipeline Schedules

| Schedule      | File                      | Status / When to use                       |
| ------------- | ------------------------- | ------------------------------------------ |
| Standard 1F1B | Megatron built-in         | PP=2 VPP=2 = 193 TFLOP/s europe (verified 2026-04-14) |
| DualPipeV     | `dualpipev_schedule.py`   | Experimental; infrastructure written but incompatible with EP>1 (DeepEP A2A cross-rank desync). `CPPMEGA_DUALPIPEV=1`, default off |
| combined_1f1b | `hybrid_schedule_plan.py` | Closed: OOM at every PP=2 MBS, PP=4 impossible (52 layers). Infrastructure kept; `CPPMEGA_EP_OVERLAP=1`, default off |

### Upstream Patches (apply_dsa_cg_patches.py)

**MUST run after every Megatron update.** 10 patches (1-9 + 9b):

1. CUDA graph safety (ban torch.equal/.any CPU syncs)
2. Remove 576/512 dim hardcodes (our dims: 128/64)
3. SparseMLA d_v propagation
4. sparse_mla.py d_v parameter
5. tilelang_sparse_mla_fwd.py dim assertions relaxed
6. tilelang_sparse_mla_bwd.py D=512 hardcode removed
7. tilelang_sparse_mla_bwd.py P/dP dtype (revert fp32 — broke TileLang GEMM dtype constraint)
8. CG-safe _scatter_topk_into_index_mask (branchless)
9. FP8 SparseMLA dispatch (QuantizedTensor → SparseMLA_FP8, zero-copy via `_data`)
9b. Remove stray `query.dequantize()` + `key.dequantize()` that had drifted into installed dsa.py (was killing zero-copy FP8 by re-quantizing BF16 tensors per-token)

### TileLang wheel

Prebuilt x86_64 wheel (cp38 stable ABI) for H200 machines lives in Google Storage:

```
sftp://BUCKET_ARTIFACTS/tilelang/tilelang-0.1.8+cuda.gitf309d814-cp38-abi3-linux_x86_64.whl
```

Install with `scripts/install_tilelang_wheel.sh` — it activates the target venv,
pulls the wheel from GS, and verifies the import. If the GS fetch fails (auth,
no network), the script falls back to cloning `tile-ai/tilelang` and building
from commit `f309d814` via `pip install -e . --no-build-isolation`.

Notes:
- Wheel works on bench3 and europe (both x86_64 H200).
- GB10 (aarch64, sm_121) must build from source — no aarch64 wheel yet.
- Build recipe: `cd tilelang-build && pip wheel . --no-build-isolation --wheel-dir /tmp/tilelang-wheel/`.

## 7. Throughput results — measured

Measurements on 8×H200 (LOCATION_2 unless noted). 25-iter median,
iters 3-25. CG must be OFF at PP=1 (TE CG private pool = 39.5 GiB).

| Config                                  | TFLOP/s | tok/sec/GPU | Notes             |
| --------------------------------------- | ------- | ----------- | ----------------- |
| **PP=1 EP=4 MBS=8 BF16 no-CG** europe   | **289** | ~9,250      | Production config |
| PP=1 EP=4 MBS=8 FP8 no-CG europe        | 279     | ~8,950      | FP8 overhead > gain |
| PP=1 EP=4 MBS=8 FP8 no-CG **bench3**    | 253     | ~8,100      | Same topology, 13% machine delta |
| PP=1 EP=4 MBS=8 FP8 +param-gather bench3 | 252    | ~8,080      | -2.6 GiB mem, neutral throughput |
| PP=2 VPP=2 EP=4 MBS=4 MTP=2 europe      | 193     | ~6,200      | Vanilla PP=2 baseline |
| PP=2 VPP=2 EP=2 DP=2 MBS=4 europe       | 193     | ~6,200      | Alternate EP split |
| combined_1f1b overlap any PP=2 config   | OOM     | —           | ~40 GiB pipe buffers + 95 GiB baseline > 141 GiB |
| combined_1f1b PP=1 MBS=4 no-MTP         | 182     | ~5,900      | -12.5% + OOM@iter 8 |

## 8. Hardware + software stack

### Machines

| Machine | Zone           | IP            | Path                     | GPU              |
| ------- | -------------- | ------------- | ------------------------ | ---------------- |
| europe  | LOCATION_2 | H200_2_IP  | `/mnt/data/cppmega-root` | 8x H200          |
| bench3  | LOCATION_1  | H200_1_IP | `/mnt/data/cppmega-root` | 8x H200          |
| GB10    | local network  | gb10          | `/home/dave`             | 1x GB10 (sm_121) |

### Megatron Version

Branch `dev_latest` on top of NVIDIA/Megatron-LM `dev`, with per-machine divergence:

| Machine | Version | Base | Our patches | Notes |
|---|---|---|---|---|
| bench3  | `megatron-core 0.18.0rc0` (installed via `pip install git+NVIDIA/Megatron-LM@980211ae`) | upstream `dev` at commit `980211ae`, 2026-04-09 | local cherry-pick of open PR #3345 + DSA integration (overlaid on tarball at `/mnt/data/cppmega-root/megatron-lm/`) | 2026-04-14 backup: `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/megatron_lm_tree.tar.gz`. **Note**: the tarball's `package_info.py` reports `0.16.0rc0` but the actually-installed site-packages is `0.18.0rc0` — see `docs/session_3_gap_audit.md` |
| europe  | `megatron-core 0.16.0rc0` | NVIDIA/Megatron-LM `origin/dev` | 2 commits ahead on local `dev_latest` branch | HEAD = `ec6a9e900`: local cherry-pick of open PR #4268 on top of local commit `2eeabc668`, which is our local merge of open PR #3674 (DSA absorbed MLA + TileLang fused sparse ops). Neither PR is merged upstream as of 2026-04-14 — see "Upstream status" below. |
| GB10    | n/a     | n/a  | local clone present (`/home/dave/megatron-lm`) + `cppmega-venv` under `/home/dave/cppmega-venv` | Primary role is bwd_bwd / TileLang / CuTe-DSL single-GPU kernel dev; full NAM56R training is not validated here (sm_121 HW caps block tcgen05/FP4/WGMMA paths). 99 KiB dynamic-smem cap is respected by every in-tree TileLang kernel via `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE: True`; enforced at training-launch by `python -m cppmega.megatron.preflight_smem_check` (hard-fail on sm_121 if any kernel is missing the flag). |

The two H200 hosts are on **different** megatron-core versions:
- bench3: `0.18.0rc0` (installed via `pip install git+...@980211ae`)
- europe: `0.16.0rc0` (editable install from `/home/dave/cppmega-root/megatron-lm/`, local `dev_latest` branch 2 ahead of `origin/dev`)

The previous README claim that "both are 0.16.0rc0" was wrong — it was based on
the `package_info.py` inside the bench3 tarball snapshot, but the *installed*
site-packages version is 0.18.0rc0. See `docs/session_3_gap_audit.md` for the
correction and `docs/megatron_restoration_recipe.md` for full base-commit
recovery + patch listing.

### Upstream status (verified 2026-04-14)

Some related upstream changes are merged, but the cppmega-critical
Megatron/FLCE/Hopper pieces we still depend on are not fully upstream or not
available in a released form. Anything referenced below lives only on our
local `dev_latest` branch until noted otherwise.

| Ref | Upstream status | What we do locally |
|---|---|---|
| Megatron-LM PR #3345 (Hopper fused linear CE) | OPEN (not merged) | Local cherry-pick on bench3 + `apply_linear_ce_patch.py` gates `CPPMEGA_PREFER_NATIVE_HOPPER_CE` on top of it |
| Megatron-LM PR #3674 (DSA absorbed + TileLang fused sparse) | OPEN (not merged) | Local merge on europe at commit `2eeabc668`; `apply_dsa_cg_patches.py` carries the additional fixes on top |
| Megatron-LM PR #4268 (delayed wgrad / P2P bwd overlap) | OPEN, **DRAFT** | Local cherry-pick on europe at commit `ec6a9e900` |
| TileLang PR #746 (bulk-copy / layout inference) | MERGED ✓ | Picked up via the pinned TileLang build `f309d814` |
| Liger-Kernel issue #968 (FLCE reduction="none" wrong grad) | CLOSED, **no functional fix shipped** | We sidestep via `reduction="mean"` broadcast workaround in `apply_linear_ce_patch.py` |
| Liger-Kernel PR #1126 (FLCE reduction="none" guard) | OPEN, **DRAFT** — assertion only, not a real fix | Not applied; we do not rely on it |
| Liger-Kernel PR #680 (LigerCrossEntropy reduction="none") | MERGED ✓, **but scope is scalar CE only** — the same commit explicitly *removes* `reduction="none"` from FLCE | Useful for scalar CE only; does not fix the FLCE path that our MTP heads hit |
| state-spaces/mamba issue #886 (Mamba3 bwd misalignment when `nheads%4 != 0` and `seqlen%4 != 0`) | OPEN (not fixed) | We keep `nheads=32` / `seqlen=4096` so the corrupt path is never taken |

### Software Stack

- PyTorch 2.12 nightly + cu132
- Transformer Engine 2.13
- TileLang 0.1.8+cuda.gitf309d814 (main-branch build; install via `scripts/install_tilelang_wheel.sh` or source)
- mamba-ssm 2.3.1
- Megatron Core: bench3 `0.18.0rc0` (pip-git@980211ae), europe `0.16.0rc0` (editable `dev_latest`). See Megatron Version above.
- NVIDIA Apex (from source)
- dualpipe 1.0.0+030ce43 (from github)
- liger-kernel

## 9. Key design decisions

- **heads=32, hidden=4096**: FP8 compatible (32%8=0), WGMMA tiling, lemyx (heads==index_heads)
- **52 layers** (NAM56R_DEPTH): divides by 4 (VPP=4) and 2 (VPP=2)
- **DSA stack** (bf16 only; FP8 indexer variants deleted 2026-04-13):
  - `lemyx_dsa_warmup.py` — lemyx/tilelang-dsa fused FA + lightning-indexer + KL kernel, used for indexer warmup (~first 1000 steps)
  - `dsa_indexer_fused_patch.py` — per-head fused accumulation replacement for Megatron's `_compute_index_scores` (bf16)
  - `dsa_splitk_indexer_loss.py` — split-K Triton fused indexer KL loss (port of NVIDIA Megatron-LM PR #4039); ~60% memory save
  - `dsa_sparse_attention.py` — sparse gather/scatter replacement for Megatron's `unfused_dsa_fn`; avoids the 7 GiB full `[b*np, sq, sk]` FP32 scores tensor
  - `dsa_sparse_absorbed.py` — same sparse-gather replacement for the absorbed-MLA DSA path (`_unfused_absorbed_dsa_fn`, PR #3674)
  - `index_cache_patch.py` — reuses indexer top-k across DSA-Shared layers (3 Full + 6 Shared pattern)
  - No try/except fallbacks in any of the above — fused kernel unavailable = crash
- **No _apply guard**: D/dt_bias stay bf16 after Float16Module, use .float() in forward
- **No silent fallbacks**: critical patches crash on failure, never try/except + continue
- **expandable_segments mandatory**: hardcoded, not overridable
- **Unfused DSA banned forever**: if fused kernel unavailable, crash immediately

## 10. Project structure

```
cppmega/
  megatron/
    index_cache_patch.py       # DSA index reuse (3 Full + 6 Shared)
    lemyx_dsa_warmup.py        # Fused FA+KL TileLang kernel
    mamba3_compile_patch.py    # Regional torch.compile (4 regions)
    dualpipev_schedule.py      # DualPipeV pipeline schedule (~470 LOC, experimental)
    hybrid_schedule_plan.py    # build_schedule_plan for MambaModel
    mtp_liger_ce.py            # Liger fused cross-entropy for MTP
    selective_fp8_moe_patch.py # FP8 only for MoE experts
    mamba_recompute_patch.py   # Mamba activation checkpointing
    noconv_mamba_mixer.py      # Mamba3 mixer (no conv1d)
    custom_mamba_model.py      # CppMegaMambaModel wrapper
    sparse_mla_ops/            # TileLang SparseMLA fwd/bwd + FP8
    upstream_patches/
      apply_dsa_cg_patches.py             # 10 patches for DSA + CG + FP8 + dims (1-9 + 9b)
      apply_mamba3_mimo_p1_patches.py     # Mamba3 TMA+warpspec (fwd only, bwd blocked upstream)
      apply_mamba3_mimo_tma_layout_fix.py # 3D→2D smem flatten, unblocks full P1 (branch: tma-layout-fix-3d-to-2d)
      apply_dualpipev_patch.py            # DualPipeV Megatron hook (experimental)
  recipes/
    nam56r_nemo_recipe.py      # NAM56R configuration
scripts/
  remote_smoke_h200_dsa_9_4_m.sh  # Production training script
docs/
  optimization_session_2026_04_13.md     # Session notes (English)
  optimization_session_2026_04_13_ru.md  # Session notes (Russian)
```
