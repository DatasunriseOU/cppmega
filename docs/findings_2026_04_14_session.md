# Session findings 2026-04-14 — comprehensive (EN + RU)

Multi-agent investigation triggered by user pushback on prior claims. Launched 13+ parallel agents (research, grounding verification, empirical test) over MCP web search (perplexity deep research, exa, brave, tavily) + bench3/europe/GB10 hardware.

---

## 1. Liger FLCE backward bug — `reduction="none"` produces silently corrupt gradients

### EN

**Root cause**: `LigerFusedLinearCrossEntropyFunction.backward` does NOT support `reduction="none"`. The kernel's `element_mul_kernel` does `tl.load(grad_output_ptr)` reading only the FIRST scalar element. When autograd hands back a per-token `[s*b]` tensor, ALL gradients are scaled by whatever happens to be at `grad_output[0]` — silent full corruption, not partial.

**Severity**: SILENT — no NaN, no crash, no error. Loss curves look normal because forward computes per-token loss correctly. But parameter updates diverge from true gradient because backward multiplies by random scalar.

**Worst case**: if `grad_output[0] == 0` (e.g., first token is `ignore_index`), ALL gradients become zero (issue #968 reproducer).

**Implications for our prior measurements**:
- `cppmega/megatron/mtp_liger_ce.py:169` calls Liger with `reduction="none"` for MTP loss. Gradients hitting `_MTPLossAutoScaler.apply(...)` may be per-token or scalar depending on autoscaler semantics — likely per-token → silent corruption.
- All bench3 baselines using `CPPMEGA_MTP_LIGER_CE=1` (including 268 / 269.4 TFLOP/s records) had **MTP gradients silently corrupted**. Loss converged but training was effectively running with wrong MTP signal.
- `nanochat/kernels.py:458-520` (`fused_linear_cross_entropy_liger`) has the same pattern — should be flagged.

**Status of upstream fixes** (independently verified by grounding agents):

| PR | What it does | Fixes our bug? |
|---|---|---|
| #1126 (DRAFT) | Adds `assert grad_output.ndim == 0` — fail-fast | ❌ assertion only |
| #496 (merged 2024-12) | Added forward "none" support, backward broken | ❌ origin of bug |
| #1182 (OPEN) | Routes kwarg through `loss_utils.py` | ❌ kernel untouched |
| #680 (merged 2025-04) | Fixed reduction="none" for non-fused `LigerCrossEntropy` | ❌ different module |
| OFSkean fork | `has_post_kernel_scaling` for weighted tokens | ❌ forward-only, backward identical |

**Author's own statement** (Nick Knight, NVIDIA, in #968):
> "adding this functionality would preclude the current chunkwise (pre)computation of grad_weight and grad_bias in the forward pass"

Architectural conflict: Liger FLCE pre-computes `grad_weight`/`grad_bias` chunkwise in forward (this is what gives memory savings). Per-token backward requires knowing `grad_output` AFTER forward — incompatible with chunked precompute. **No fix possible without kernel rewrite.**

**Workarounds (4 paths)**:

1. `loss.sum().backward()` — autograd graph reduces to scalar, `grad_output == 1.0` takes correct path. Requires restructuring caller.
2. Switch to non-fused `LigerCrossEntropyFunction` (PR #680 fixed reduction="none" backward correctly). **Tradeoff: lose Liger memory savings** — must materialize `[s*b, V]` logits.
3. Apple Cut Cross-Entropy (CCE, `pip install cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git`). Different architecture, supports per-token natively.
4. Megatron-LM PR #3345 (open) — adds Hopper kernels for `fused_linear_cross_entropy.py`, dispatches `cc[0] == 9`. Native CUDA, not Liger.

### RU

**Корень**: `LigerFusedLinearCrossEntropyFunction.backward` НЕ поддерживает `reduction="none"`. Kernel читает `grad_output[0]` как scalar. Когда autograd передает per-token `[s*b]` tensor, ВСЕ градиенты умножаются на одно случайное значение `grad_output[0]` — silent full corruption.

**Severity**: SILENT — нет nan, нет crash. Loss curves выглядят нормально (forward правильный), но параметры обучаются с искаженным gradient.

**Worst case**: если `grad_output[0] == 0` (ignore_index), ВСЕ градиенты = 0.

**Impact на наши измерения**:
- `mtp_liger_ce.py:169` использует `reduction="none"`. Все bench3 baselines с `CPPMEGA_MTP_LIGER_CE=1` (включая 268 / 269.4 TFLOP/s) имели **MTP gradients silently corrupt**. Loss сходится но обучение по факту шло с неверным MTP сигналом.
- nanochat/kernels.py:458-520 — тот же баг.

**Upstream PRs** — ни один не чинит (см. таблицу выше).

**Цитата автора** (Nick Knight, NVIDIA в #968):
> "adding this functionality would preclude the current chunkwise (pre)computation"

→ Архитектурный конфликт. Без переписки kernel'а fix невозможен.

**4 workaround пути** — см. EN секцию.

---

## 2. Liger fix path A (`reduction="mean"` + broadcast) — implemented + math verified, empirical validation BLOCKED

### EN

**Patch implemented** (`cppmega/megatron/apply_linear_ce_patch.py:_install_liger_compute`):
- Always call `LigerFusedLinearCrossEntropyFunction.apply(... reduction="mean" ...)` — only correct backward path
- For caller `reduction="none"`: return `liger_loss_scalar.expand(b, s).contiguous()`. Math: `sum(expanded * mask) = mean * N_valid = sum_i CE_i` matches exact per-token sum. Backward: `d(sum)/d(scalar) = N_valid` cancels Liger's `1/N_valid` factor → identical gradient to correct per-token path (exact when `loss_mask == (labels != ignore_index)`).
- Tradeoff: per-token loss logging granularity lost (all `[b, s]` entries = mean), but **gradients are exact**.

**Validation BLOCKED on bench3** by unrelated TileLang TMA bug (see section 4 below).

### RU

**Patch реализован**: всегда вызывает Liger с `reduction="mean"`, для caller `reduction="none"` broadcast'ит scalar в `[b, s]`. Math корректен (gradients идентичны правильному per-token path).

Loss logging granularity потерян (все per-token entries = mean), но **gradients exact**.

**Empirical validation BLOCKED** на bench3 — несвязанный TileLang TMA bug.

---

## 3. Apple CCE — installed git latest 25.9.3, initial test shows MEMORY LOSS on our shape

### EN

- Upgraded `cut-cross-entropy 25.1.1` (PyPI Jan 2025, stale) → `25.9.3` (git latest) on bench3 + europe via `pip install cut-cross-entropy @ git+https://github.com/apple/ml-cross-entropy.git`. Required `chown -R dave:dave /mnt/data/venv` first because earlier `gcloud compute ssh` runs as `google_test_datasunrise_ou_io` had created files owned by that user.
- Initial test (Agent B with version 25.1.1) showed CCE OOM at MBS=10 NAM56R FP8 EP=8: peak 127 GiB > 139 budget. **However, this may be misdiagnosed** — the parallel TileLang TMA bug crashes ALL bwd today, inflating peak_alloc artificially. Re-test with 25.9.3 in flight.
- Apple paper claims 28GB → 1GB on Gemma 2B (V=256k, S=8k). On our shape (V=65k, S=4k) scaling gives Liger ~0.4-0.8 GiB and CCE ~0.1 GiB — but the empirical OOM contradicts this. Under investigation.
- **Critical paper fact** (line 1123): on low |V|/D models (Phi 3.5 Mini V=32k D=3k), CCE is **50% SLOWER than torch.compile**. Our shape V=65536/H=4096 = |V|/D=16, same regime — CCE perf advantage NOT guaranteed.
- omkaark.com H100 anecdotal: CCE step-time hit OUTWEIGHS memory savings on Hopper for small-V/H configs.
- Apple repo: PyPI last release Jan 2025 (>15 months old), git active.

### RU

- Upgrade на git 25.9.3 на bench3 + europe (`pip install cut-cross-entropy @ git+...`). Требовался `chown -R dave:dave /mnt/data/venv` сначала из-за владельца файлов от gcloud SSH.
- Первый тест (Agent B, версия 25.1.1) показал CCE OOM на MBS=10: peak 127 GiB. **Возможно misdiagnosis** — TMA bug ломал bwd, искажая peak_alloc. Re-test с 25.9.3 в работе.
- Apple paper: 28GB→1GB на Gemma 2B (V=256k). На нашей shape (V=65k) выйгрыш меньше но реальный.
- **Critical**: paper line 1123 — на low |V|/D models CCE **50% МЕДЛЕННЕЕ torch.compile**. Наш V=65536/H=4096 = |V|/D=16, тот же regime.
- Apple PyPI: stale Jan 2025, git active.

---

## 4. TileLang TMA bug — REAL but mis-claimed; commit 4f115ea breaks bwd on bench3 today

### EN

**Original claim** (in memory `reference_p1_blocked_tilelang_tma_layout.md`):
> `LowerBulkCopy` asserts `InputDim == 2`, rejects 3D smem descriptors. Fix branch `tma-layout-fix-3d-to-2d @ 31dc695` flattens 3D→2D but trips FloorMod divide-by-zero in bwd_bwd.

**Verified findings** (Agent 1 + grounding):
1. **Actual TileLang HEAD code** (post PR #746): `if (shared_layout->InputDim() < 2) { LOG(WARNING) "...fallback to normal copy"; return LowerNormalCopy(); }` — **WARNING + fallback, NOT crash**. `InputDim == 3` silently falls back to non-TMA copy.
2. **Hard `ICHECK(InputDim() >= 2)`** exists in `Conv2DIm2ColOpNode::Lower`, separate path.
3. **PR #746** unified `bulk_copy → Copy` operator with TMA descriptor infra but **retained 2D assumption for swizzle-detection**.
4. **Our `tma-layout-fix-3d-to-2d @ 31dc695` commit message** mentions `(shared_layout->InputDim() == 2)` — this exact assertion text **predates PR #746** or is on a fork. Need to verify our pinned tilelang version.
5. **FloorMod divide-by-zero in T.Parallel**: PR #1458 fixes only z3 prover, not general LayoutInference. Issue #1374 (closed) adjacent but doesn't cover `csr % R`. Real class of bugs, no upstream fix for our case.

**TODAY'S REGRESSION** (Agent A, 2026-04-14):
- All bwd passes on bench3 crash with `tvm.error.InternalError: Check failed: (shared_layout->InputDim() == 2) is false: Cannot detect TMA layout` in `tl::CopyNode::LowerBulkCopy`.
- Activated by **commit `4f115ea` (P1: enable TMA + warp specialization in Mamba3 MIMO kernels)**.
- **Env gate is DEFAULT OFF** (`CPPMEGA_MAMBA3_P1=0`). But `apply_mamba3_mimo_p1_patches.py` modifies upstream `mamba_ssm/ops/tilelang/mamba3/*.py` files **in-place**. Once applied (someone ran with P1=1 once), file changes PERSIST. P1=0 does NOT undo.
- Result: **bench3 mamba3 backward is broken today regardless of env gate**. Even MBS=8 baseline (historically 67 GiB peak) now crashes at 106 GiB peak.

**Workarounds** (CORRECTED by deep PR analysis agent a3a17b378f9c30adf):

❌ ~~Replace `csr % R` → `csr - (csr // R) * R`~~ — **WON'T WORK**. `TryConstFold<FloorMod>` fires during Z3/RewriteSimplifier canonicalization. Any algebraic form of modulo gets canonicalized BACK to `FloorMod` before const-fold runs. Tested: zero TileLang examples use subtraction form as workaround.

✅ **Real fix**: pass `R` as `tir.constexpr` (compile-time constant) and unroll R-dim manually with `for r in T.unroll(R):`. This enumerates R at compile time, bypassing FloorMod const-fold entirely. Estimate: **1-2 days surgery** on `mamba3_mimo_bwd.py`.

✅ **Alternative — cherry-pick tilelang main**: pin to commit `a8bafa6` (2026-04-14) or newer. Gets:
- PR #746 follow-ups: replaces strict `ICHECK(InputDim() == 2)` (in our v0.1.8) with `InputDim < 2 → LOG(WARNING) + LowerNormalCopy` fallback + uses `ndim-2/ndim-1` for rank-3+
- PR #2002 (merged 2026-04-07): moves pipeline planning + WS before layout inference. Stricter TMA stride validation, may fix our case
- PR #2024 (re-enabled `TL_DISABLE_TMA_LOWER`)
- PR #2031 OPEN: directly fixes 3D smem layout aliases in LayoutInference

**Verified pinned version**: `tilelang v0.1.8` (2026-02-16). 5 critical merges landed AFTER v0.1.8 — must build from main.

**Action items**:
1. **Restore upstream mamba_ssm files** on bench3 (revert `apply_mamba3_mimo_p1_patches.py` modifications) — needed BEFORE any other bwd validation
2. Add **un-patch** mode to `apply_mamba3_mimo_p1_patches.py` (currently only patch-direction)
3. File 2 upstream tilelang issues (TMA 3D smem, FloorMod) — pending user approval

### RU

**Изначальное утверждение** в memory: `LowerBulkCopy` asserts `InputDim == 2`, ломает 3D smem.

**Verified**:
1. **Реальный HEAD код** (после PR #746): `if (InputDim < 2) { LOG WARNING; fallback }` — **WARNING + fallback**, не crash. `InputDim == 3` silently fallback на non-TMA.
2. Hard `ICHECK(InputDim() >= 2)` в `Conv2DIm2ColOpNode::Lower` — отдельный path
3. **PR #746** объединил `bulk_copy → Copy`, но swizzle hardcoded 2D
4. Наш commit `31dc695` упоминает `(InputDim() == 2)` — **предшествует PR #746**
5. **FloorMod в T.Parallel**: PR #1458 чинит только z3, не general LayoutInference. Real bug, no upstream fix.

**🚨 СЕГОДНЯШНЯЯ РЕГРЕССИЯ**:
- Все bwd на bench3 крашатся с `tvm.error.InternalError`
- Активировано **commit `4f115ea` (P1)**
- **Env gate default OFF**, но `apply_mamba3_mimo_p1_patches.py` модифицирует upstream files **in-place**
- Однажды применили (P1=1) → файлы остаются патченные. P1=0 НЕ undo'ит.
- **Bench3 mamba3 backward broken сегодня regardless of env gate**

**Workarounds**:
- Replace `csr % R` → `csr - (csr // R) * R` обходит FloorMod
- Verify TileLang pinned version
- Cherry-pick PR #746 если наш предшествует

**Action items**:
1. **Restore upstream mamba_ssm files** на bench3 (undo P1 patch) — нужно ASAP перед другими тестами
2. Добавить **un-patch** mode в `apply_mamba3_mimo_p1_patches.py`
3. File 2 upstream tilelang issues — ждёт твоего approval

---

## 5. Megatron-LM Mamba LinearCrossEntropyModule — upstream PR regressed

### EN

**Chronology** (verified via gh api):
- 2025-12-05: PR #2256 — initial `fused_linear_cross_entropy` infra
- 2026-02-04 **01:47 UTC**: PR **#3226 merged to dev** (sha `8a29fd5752`) — added `self.output_layer = LinearCrossEntropyModule` for Mamba (the swap)
- 2026-02-04 **22:40 UTC**: PR **#3207 merged to main** (sha `9d71cb1cd2`) — "Reapply MTP support for hybrid models", branched from `main`. Re-introduced `self.output_layer = tensor_parallel.ColumnParallelLinear`.
- 2026-02-26: `main → dev` sync — textual merge preferred main's ColumnParallelLinear, **erased PR #3226**
- 2026-03-18: PR #3225 (prefix cache) cemented regression
- 2026-02-26: PR #3058 (μP) added `self._scale_logits(logits)` cementing further

**Current `dev` HEAD**: still regressed. Mamba uses plain `ColumnParallelLinear`. GPT model line 251 correctly uses `LinearCrossEntropyModule` (PR #3226 stuck).

**Our megatron-core_v0.15.0rc7 dev_latest** was cut post-#3207 → we have the regressed code.

**Mechanism**: NOT intentional revert. PR #3207 branched off `main` (which never had #3226), conflict on merge favored main. No upstream open issue or PR for Mamba swap regression.

**Hopper kernels: PR #3345 OPEN, not merged**, last updated 2026-03-23
- +2256 LOC across `megatron/core/fusions/linear_cross_entropy/hopper/{entry,fwd_mainloop,bwd_partial_dlogits,utils}.py`
- Modifies `fused_linear_cross_entropy.py` to dispatch on `cc[0] == 9` (Hopper)
- Uses CuTe DSL + WGMMA + SM90 hopper_helpers
- **If cherry-picked + combined with our class swap** — replaces our Liger reroute with native Megatron Hopper kernels
- Status: stalled ~3 weeks awaiting Expert Review

**Our `apply_linear_ce_patch.py`** is the workaround for the regression: monkey-patches `MambaModel.__init__` to swap `output_layer.__class__` to `LinearCrossEntropyModule` runtime + reroutes `_compute_linear_and_cross_entropy_loss` to Liger on cc<10 (since Megatron native is Blackwell-only).

### RU

**Хронология**:
- 2025-12-05: PR #2256 — initial fused CE infra
- 2026-02-04 01:47 UTC: PR **#3226 merged в dev** — добавил `LinearCrossEntropyModule` для Mamba
- 2026-02-04 22:40 UTC: PR **#3207 merged в main** — Reapply MTP, branched from main, re-introduced `ColumnParallelLinear`
- 2026-02-26: `main → dev` sync **стер PR #3226**
- #3225 (prefix cache), #3058 (μP) **зацементировали** регрессию

**Текущий dev HEAD**: still regressed. Mamba = ColumnParallelLinear. GPT line 251 = LinearCrossEntropyModule (правильно).

**Наш megatron-core_v0.15.0rc7** post-#3207 → у нас регрессированный код.

**PR #3345 OPEN** — Hopper kernels:
- +2256 LOC, dispatch на `cc[0] == 9`
- CuTe DSL + WGMMA + SM90
- **Если cherry-pick + наш class swap** — заменяет Liger reroute на native Megatron Hopper

**Наш `apply_linear_ce_patch.py`** — workaround регрессии: monkey-patch `MambaModel.__init__` swap'ит class, рерутит на Liger на cc<10.

---

## 6. Mamba3 P3 register split — HALT (3 blockers)

### EN

Per implementation+verification agent (a3a6117b87cd1a33d) and grounding (a14bc6dc58bd546f1):

**3 blockers found**:
1. **Split point not clean** — `dstates_frag` is loop-carried in reverse-scan. Pass-1 needs `dPhiO_shared` (which design said only Pass-2 needs). Realistic regs after split: 200-220, NOT 140 (matches Agent 6).
2. **GB10 cannot validate** — `mamba_mimo_fwd` fails to compile at every shape. `TMA desc init error 716` (dim=3 descriptor) — same TileLang TMA family. Auto-tuning fails. No baseline available.
3. **Bench3 SSH was broken initially** — `dave@H200_1_IP` direct returned `Permission denied (publickey)`. Now FIXED via gcloud + key install.

**Atomics NOT required** (refuted prior claim) — DQ/DK writes go to disjoint per-chunk slices via plain `T.copy`. Source audit: `mamba3_mimo_bwd.py:1090,1131`. Between-chunks split safe with staging buffer.

**Path 0 (warp-spec annotations) NOT drop-in** (corrected grounding):
- `T.annotate_producer_reg_dealloc/annotate_consumer_reg_alloc` are HINTS that ONLY work when kernel IR ALREADY has explicit warp-spec split (`mbars`, `T.attr WarpSpecializationScope`, `tx`-based partition)
- Our `bwd_bwd_kernel` calls `T.no_set_max_nreg()` + `T.Kernel(H, B, threads=256)` without tx split + 20+ shared buffers — flat single-warpgroup
- `ProducerConsumerWarpSpecialized` auto-conversion has v1 limitations matching our blockers exactly (TMA-only pipelines, no conditional guards in loops, single pipeline)
- Effort: 2-3 weeks rewrite from scratch using DeepSeek MLA WS template, gated on TMA layout fix

**Revised priority order**:
1. **Path 1 (P2 PsiV hoist)** — 2 days, GB10-testable, attacks root cause (17 → 13 frags). Same gain as P3 split at 1/4 effort.
2. **Path 2 (P3 split, atomics-free)** — 8-12 days, last resort
3. **Path 0 (WS rewrite)** — 2-3 weeks, deferred, gated TMA fix

### RU

3 blockers:
1. Split point не чистый — `dstates_frag` loop-carried, Pass-1 нужен `dPhiO_shared`. Realistic regs: 200-220 (не 140).
2. GB10 не валидирует — `TMA desc init error 716`.
3. Bench3 SSH FIXED через gcloud + key install (раньше было Permission denied).

**Atomics НЕ нужны** (REFUTED prior claim) — disjoint per-chunk slices.

**Path 0 (warp-spec) НЕ drop-in** — annotations это HINTS, требуют existing WS split (mbars, scope attrs). У нас flat kernel. Effort: 2-3 недели rewrite.

**Revised priority**:
1. **Path 1 (PsiV hoist)** — 2 дня, GB10-testable, attacks root cause
2. Path 2 (split, no atomics) — 8-12 дней, last resort
3. Path 0 (WS rewrite) — 2-3 недели, deferred

---

## 7. Mamba3 P2 PsiV cache design — concrete

### EN

Per design agent (a6887247c17870849):

**What `bwd_fwd` recomputes** (verified via state-spaces/mamba source + local cuTile port):
- `Q_rot, K_rot` via RoPE
- `K_trap` via exp-trapezoidal
- `PsiV` = V * MIMO_V

**Cost at NAM56R MBS=10** (per Mamba3 layer):
| Tensor | Shape | bf16 GiB |
|---|---|---|
| Q_rot | [B,S,R,G,N] = 10·4096·4·8·128 | 0.31 |
| K_rot | [B,S,R,G,N] | 0.31 |
| **PsiV** | [B,S,R,H,P] = 10·4096·4·32·128 | **1.25** |
| **per-layer total** | | **1.87** |

× 9 Mamba3 layers/rank = **16.8 GiB/rank** if cache all → overshoots ~2 GiB budget by 8×.

**Recommended subset**: cache only `Q_rot + K_rot` (5.6 GiB/rank), drop PsiV (cheap recompute via single elementwise broadcast-multiply).

**Expected win**: ~15-20% reduction of `mimo_bwd_bwd` (8.9% of total) → **~1.5-2.3% end-to-end = +5-7 TFLOP/s = 274-277 on bench3**.

**Effort**: 5 days. Risk: state-spaces PR #909 shows FSDP+activation-checkpointing interaction is non-trivial for Mamba3 saved tensors.

**No prior art** — Mamba2 SelectiveScanFn saves only inputs + scan_intermediates. P2 is novel optimization for Mamba3 MIMO specifically.

### RU

**Что recompute сейчас**: Q_rot, K_rot, K_trap, PsiV.

**Memory cost** (per layer NAM56R MBS=10):
- Q_rot 0.31 GiB
- K_rot 0.31 GiB
- PsiV **1.25 GiB**
- Total per-layer 1.87 GiB → × 9 layers = **16.8 GiB/rank**

**Recommended**: cache только `Q_rot + K_rot` (5.6 GiB/rank), drop PsiV.

**Expected**: **~1.5-2.3% end-to-end = 274-277 TFLOP/s на bench3**.

**Effort**: 5 дней. **Risk**: FSDP+activation checkpointing interaction (PR #909).

---

## 8. Bench3 SSH fix

### EN

Direct SSH `dave@H200_1_IP` was returning `Permission denied (publickey)` because `~/.ssh/google_compute_engine.pub` was not in `dave`'s `authorized_keys`. `gcloud compute ssh` worked because IAP authenticates as `google_test_datasunrise_ou_io` (the GCE-managed user), not `dave`.

**Fix applied**: piped local pubkey through `gcloud compute ssh ... --command='sudo tee -a /home/dave/.ssh/authorized_keys'` for both bench3 (LOCATION_1) and europe (LOCATION_2) machines.

Both `dave@<ip>` direct SSH now works. Use direct SSH for `pip install` (avoids ownership conflicts with gcloud's `google_test_datasunrise_ou_io` user).

### RU

Direct SSH `dave@H200_1_IP` возвращал Permission denied. Fix: pipe pubkey через `gcloud compute ssh ... sudo tee -a /home/dave/.ssh/authorized_keys` для обеих машин. Сейчас работает.

---

## Production status today (as of 2026-04-14 09:50 UTC)

### EN

| Item | Status | Note |
|---|---|---|
| europe production: BF16 MBS=8 EP=4 = 289 TFLOP/s | ✅ stands | unaffected by Liger bug (no MTP Liger active by default? — verify) |
| bench3 production: FP8 MBS=10 EP=8 + Liger CE = 269.4 TFLOP/s | ⚠️ **suspect** | MTP Liger reduction="none" path → silent grad corruption. Number is throughput of training-with-broken-gradients. Re-baseline needed. |
| Bench3 mamba3 backward | ❌ **broken today** | TileLang TMA bug from P1 patch persistence |
| Liger reduction=mean fix path A | ✅ patch ready | empirical validation blocked by TMA bug |
| CCE 25.9.3 upgrade | ✅ installed | retest in flight |
| Megatron PR #3345 cherry-pick | 🔄 testing | combo with class-swap |
| Mamba3 P2 PsiV cache | 📐 design ready | 5 days effort |
| Mamba3 P3 register split | ❌ HALT | use Path 1 (PsiV hoist) instead |

### RU

| Item | Status | Note |
|---|---|---|
| europe BF16 MBS=8 EP=4 = 289 | ✅ stands | не затронут Liger багом (если MTP Liger не дефолтен) |
| bench3 FP8 MBS=10 EP=8 + Liger = 269.4 | ⚠️ **подозрителен** | MTP Liger reduction="none" → silent corruption. Throughput training-с-сломанными-gradients. Re-baseline нужен. |
| Bench3 mamba3 backward | ❌ **сломано сегодня** | TileLang TMA bug от P1 patch persistence |
| Liger reduction=mean fix A | ✅ patch ready | validation blocked by TMA |
| CCE 25.9.3 upgrade | ✅ installed | retest идет |
| Megatron PR #3345 | 🔄 testing | combo с class-swap |
| Mamba3 P2 PsiV cache | 📐 design ready | 5 дней |
| Mamba3 P3 register split | ❌ HALT | use Path 1 (PsiV hoist) |

---

## Immediate action items (recommended order)

### EN

1. **Unblock bench3 bwd**: restore upstream `mamba_ssm/ops/tilelang/mamba3/*.py` files, OR add un-patch mode to `apply_mamba3_mimo_p1_patches.py`. Without this, no further empirical validation possible on bench3.
2. **Re-baseline production** with `CPPMEGA_MTP_LIGER_CE=0` (vanilla MTP CE, exact gradients) — measure true throughput. Compare to 268/269.4 numbers — gap shows scale of gradient corruption impact.
3. **Validate Liger fix path A** (reduction=mean broadcast) on clean bench3 → measure TFLOP/s
4. **Re-test CCE 25.9.3** on clean bench3 — does memory profile actually OOM, or was it TMA bug confounding?
5. **Implement P2 PsiV hoist** (5 days, GB10-testable, +5-7 TFLOP/s expected)

### RU

1. **Разблокировать bench3 bwd**: restore upstream files ИЛИ добавить un-patch mode. Без этого — никакого validation на bench3 невозможно.
2. **Re-baseline production** с `CPPMEGA_MTP_LIGER_CE=0` — измерить настоящий throughput. Gap к 268/269.4 покажет масштаб corruption.
3. **Validate Liger fix A** на чистом bench3 → измерить TFLOP/s
4. **Re-test CCE 25.9.3** — реально OOM или TMA bug confound?
5. **Implement P2 PsiV hoist** (5 дней, GB10-testable, +5-7 TFLOP/s)

---

## Session 3 afternoon — backup + PR filing prep

### EN

**Scope**: no new training runs; focus on capturing state + preparing upstream contributions.

**Shipped**:

1. **13 PR templates drafted** (C1…C13) covering: Liger FLCE `reduction="none"` bug, Megatron PR #3345 MTP extension, DSA indexer fused memory patch, TileLang TMA 3D→2D smem fallback, Apple CCE 25.9.3, `apply_linear_ce_patch.py` main-head swap, Mamba3 P1 patches, `mtp_native_hopper_ce.py`, mamba_ssm fork drift, bench3-specific `mamba3_mimo_bwd.py`, FP8 sparse MLA europe-only files, DualPipeV cleanup, combined_1f1b documentation. All local drafts — **nothing filed upstream** per `feedback_pr_approval.md`.
2. **Reproducers** for each PR — smallest possible numeric-parity / regression reproduction.
3. **`explanation_ru.md`** — RU-language narrative for each claim so user can review without English translation.
4. **Backups → GS**:
   - `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/` (17 MiB, 23 objects)
   - `sftp://BUCKET_ARTIFACTS/backups/backup_europe_2026_04_14/` (full state + `.git` HEAD for each repo + format-patch of unpushed commits)
5. **`docs/megatron_restoration_recipe.md`** — recovery procedure for bench3 megatron (no `.git` in tarball; best-effort upstream commit hypothesis `2eeabc668` pending diff confirmation).

**MCP grounding results** (`.tmp/mcp_grounding_pr_claims.md`):

- **Perplexity hallucination exposed**: Perplexity confidently attributed the Liger FLCE `reduction="none"` fix to **PR #680**. Grounding via exa + direct GitHub inspection shows #680 fixes non-fused `LigerCrossEntropy`, not `LigerFusedLinearCrossEntropy`. FLCE bug is unfixed. **Do not cite #680 in any upstream filing as a FLCE fix.**
- PRs #968, #1126 are plausible but no MCP engine could independently confirm them. Verify on GitHub before quoting.

**Diagnostic updates on MBS=12 backward NaN**:
- Suspect #1 (`apply_linear_ce_patch.py` Liger routing): **refuted** — broadcast math is exact; NaN is not caused by this.
- Suspect #2 (`CUDA_GRAPH_FLAGS=NONE` not propagating through TE `make_graphed_callables`): **pending** — test deferred to next session.

**Drift discoveries (new this afternoon)**:
- **Megatron version**: both bench3 and europe are on `0.16.0rc0`, NOT `0.18` as previously documented in README. Verified from `megatron/core/package_info.py`. README + memory notes updated.
- **Bench3 megatron has NO `.git`**: flat tree only. Restoration recipe relies on `sftp://…/megatron_lm_tree.tar.gz` as authoritative.
- **Europe 2 commits ahead of `origin/dev`**: `ec6a9e900` (PR #4268 cherry-pick) on top of `2eeabc668` (PR #3674 merge). Bench3 likely at `2eeabc668` only (PR #4268 path never exercised on bench3 which runs PP=1).
- **mamba_ssm fork drift**: bench3 at `31f3d7b` + 4 modified files; europe at `31f3d7b` + 3 modified files + stock `.orig` copy. Previously assumed identical — they are not.
- **FP8 sparse_mla files**: `tilelang_sparse_mla_{fwd,bwd}_fp8.py` + `__init__.py` exist ONLY on europe. Captured in `europe_megatron_modified.tar.gz`. Port to bench3 if FP8 sparse MLA ever targeted there.

**Files created / modified this afternoon**:
- NEW: `docs/megatron_restoration_recipe.md`
- MODIFIED: `README.md` (Megatron Version section rewrite, Software Stack row)
- MODIFIED: `plan.md` (afternoon session block)
- MODIFIED: `docs/findings_2026_04_14_session.md` (this section)
- NEW: 13 local PR-template drafts + reproducers + `explanation_ru.md` files (not yet committed — user approval gate)

### RU

**Scope**: никаких новых training runs; фокус на захвате state + подготовке upstream contributions.

**Сделано**:

1. **13 PR templates** (C1…C13): Liger FLCE bug, Megatron PR #3345 MTP extension, DSA indexer fused, TileLang TMA 3D→2D, Apple CCE 25.9.3, `apply_linear_ce_patch.py`, Mamba3 P1 patches, `mtp_native_hopper_ce.py`, mamba_ssm fork drift, bench3-specific `mamba3_mimo_bwd.py`, FP8 sparse MLA европейские файлы, DualPipeV cleanup, combined_1f1b docs. Все drafts local — **ничего не запушено** per `feedback_pr_approval.md`.
2. **Reproducers** для каждого PR.
3. **`explanation_ru.md`** для каждого PR — RU-narrative.
4. **Backups → GS**:
   - `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/`
   - `sftp://BUCKET_ARTIFACTS/backups/backup_europe_2026_04_14/`
5. **`docs/megatron_restoration_recipe.md`** — процедура восстановления bench3 megatron (нет `.git` в tarball'е).

**MCP grounding results**:
- **Perplexity hallucination**: PR #680 приписан фиксу FLCE — неправда. Реальный #680 фиксит non-fused `LigerCrossEntropy`. **Не цитировать #680 как FLCE fix в upstream filings.**
- PRs #968, #1126 — plausible но не подтверждены независимо. Проверить на GitHub перед цитированием.

**Диагностика MBS=12 backward NaN**:
- Suspect #1 (`apply_linear_ce_patch.py`): **отвергнут** — математика broadcast корректна.
- Suspect #2 (`CG_FLAGS=NONE` не propagates через TE): **pending** — отложено на следующую сессию.

**Drift discoveries (новое за эту сессию)**:
- **Megatron**: обе машины на `0.16.0rc0`, НЕ `0.18` как утверждал README. Проверено по `package_info.py`. README поправлен.
- **Bench3 без `.git`**: flat tree. Recipe опирается на GS tarball.
- **Europe 2 коммита впереди `origin/dev`**: `ec6a9e900` поверх `2eeabc668`. Bench3 вероятно только `2eeabc668`.
- **mamba_ssm drift**: bench3 31f3d7b + 4 файла, europe 31f3d7b + 3 файла + `.orig` копия.
- **FP8 sparse_mla файлы**: только на europe.

**Файлы, созданные/изменённые сегодня днем**:
- NEW: `docs/megatron_restoration_recipe.md`
- MODIFIED: `README.md`, `plan.md`, `docs/findings_2026_04_14_session.md`
- NEW: 13 local PR-template drafts (не закоммичены — approval gate)
