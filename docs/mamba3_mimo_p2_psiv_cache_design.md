# Mamba3 MIMO P2: PsiV Cache Design (Scaffolding, Not Active)

**Plan**: `plan.md` §50-53, `reference_mamba_ssm_optimization_plan.md` — P2 (2-5 days, +1.5-2.3% TFLOP/s).
**Status**: design + skeleton only (2026-04-14). Not active. Env gate `CPPMEGA_MAMBA3_P2_PSIV_CACHE=1`.
**Depends on**: `apply_mamba3_mimo_p1_patches.py` infra (same patch-import pattern).
**Cross-ref**: `docs/mamba3_mimo_p3_register_split_design.md` — P3 blocker 1 identified "Hoist-PsiV" (this P2) as the preferred path.

---

## 1. What "PsiV" is, why it is recomputed every kernel pass

In the upstream Mamba3 MIMO kernels (`mamba_ssm/ops/tilelang/mamba3/mamba3_mimo_*.py` — mirrored in our cuTile port at `.tmp/mamba3_mimo_cutile/`), the "up-projected V" tensor **PsiV** has shape `(CHUNK, R, P)` and is defined per chunk, per head, per batch as

```
psi_v[cs, r, p] = v[b, chunk_start+cs, h, p] * psi[h, r, p]
```

where `psi` is the learned per-head (`MIMO_V`) projection tensor of shape `(H, R, P)` (for NAM56R: `H=16, R=4, P=64`).

It appears **five times** inside the kernel loop body:

| Kernel | Uses PsiV in |
|---|---|
| `mamba_mimo_fwd`     | intra-chunk qk·PsiV mma (line 252), diag qk·PsiV mma (267), interchunk state accumulation (329) |
| `mamba_mimo_bwd_fwd` | recomputes `psi_v = v * psi` at line 107 — identical to fwd |
| `mamba_mimo_bwd_bwd` | recomputes `psi_v_3d_bf = v_bc_r * psi_bc_cs` at line 231 + uses `psi_bf` directly in dv_pre/dPsi_pre |

So in a full training step, for **every** chunk × head × batch, the kernel loads `V[b, c, h]` and `psi[h]` from gmem, broadcasts them, multiplies, and discards — three times. The product is not saved across kernels; each kernel rematerialises from first principles.

The P3 blocker 1 write-up (2026-04-14) independently identified this as a meaningful source of register pressure inside `bwd_bwd` — specifically, **3 fragment tiles held live** across the inner loop for the state-reverse scan.

---

## 2. Dependency analysis: **static vs dynamic inputs**

The question that makes or breaks a cross-kernel cache: **does PsiV depend only on sequence-static params, or does it depend on per-step hidden state?**

### PsiV's inputs

| Symbol | Shape | Static (per-batch/per-seq) | Changes across forward call? |
|---|---|---|---|
| `psi` (= `MIMO_V` param) | `(H, R, P)` | Yes — **module parameter** | No within a step; Yes across optim step |
| `v` (= up-projected V) | `(B, S, H, P)` | No — **derived activation** | Yes every forward (function of hidden state, in_proj weights, etc.) |

**Verdict**: PsiV is a point-wise product of a **module parameter** (`psi`) and a **hidden-state-derived activation** (`v`). It **cannot be cached as-a-whole across training steps** because `v` changes every forward pass.

### What CAN be cached within one step

Per-step, PsiV **IS** a well-defined tensor of shape `(B, S, H, R, P)` that fwd, bwd_fwd, and bwd_bwd all recompute from the same inputs. Within a single forward+backward iteration:

1. `mamba_mimo_fwd` is called during forward — computes PsiV internally.
2. `mamba_mimo_bwd_fwd` is called during backward (stage 1) — recomputes PsiV.
3. `mamba_mimo_bwd_bwd` is called during double-backward (stage 2) — recomputes PsiV a third time.

**The P2 gain is *intra-step* not *inter-step*.** Save PsiV during `fwd` (or its recompute in `bwd_fwd`), hand it to `bwd_fwd` and `bwd_bwd` as an extra kernel input, skip the recompute there.

This aligns with the `p3` design doc's "Hoist-PsiV alternative" recommendation (P3 doc line 185-189): *"cache post-rotary Q/K + PsiV from `bwd_fwd` into gmem, consume them in `bwd_bwd`. This removes ~3 frag tiles from `bwd_bwd`'s inner live-set without touching the reverse-scan."*

### Mixed dependency summary

| PsiV usage | Cacheable? | Why |
|---|---|---|
| Across training steps | **No** | `v` is a changing activation, not a param |
| fwd → bwd_fwd (same step) | **Yes** | Both read the same `V` tensor — save PsiV from fwd to gmem, load in bwd_fwd |
| bwd_fwd → bwd_bwd (same step) | **Yes** | bwd_bwd gets bwd_fwd's saved tensors for gradient checkpointing anyway; adding PsiV is small |
| Across CUDA graph replays | **No** | Activation tensor addresses are static under CUDA graphs, but cached values would be from the PREVIOUS replay. Must be recomputed every replay |

**Bottom line: P2 cache = intra-step gmem checkpoint tensor, NOT a cross-step hash/dict cache.**

---

## 3. Cache key design

Not applicable in the classical hash-of-inputs sense.  The "cache" is a gmem tensor slot the Python autograd wrapper allocates per-call and passes to the bwd kernels alongside the existing saved-for-backward tensors.  The identity of the tensor is determined by the autograd op's `saved_tensors` list — no hash needed.

**But**: if the Python wrapper decides to *reuse* the same gmem buffer across autograd ops (e.g. pre-allocated pool), then it must key on:
- Shape: `(B, S, H, R, P)`
- Dtype: `V.dtype` (typically `bf16`)
- Layout: chunk-contiguous is preferable for bwd access pattern (loads `v[c, :, i_h, :]` per chunk)

For the first implementation, we will **not** pool buffers — just `torch.empty()` one fresh per forward call and let PyTorch autograd's save-for-backward machinery hold the reference until `bwd_bwd` completes.

---

## 4. Cache storage

**Option A — activation saved-for-backward (CHOSEN)**
- Autograd `ctx.save_for_backward(...PsiV_cache_tensor...)` inside `cppmega_tilelang_mimo_combined.forward`.
- Kernel signature for `mamba_mimo_fwd` gets an extra **output** tensor `PsiV_out` (shape `(B, S, H, R, P)`, dtype `V.dtype`).
- `mamba_mimo_bwd_fwd` and `mamba_mimo_bwd_bwd` get an extra **input** tensor `PsiV_in` and skip the `psi_v = v * psi` recompute.
- Zero-cost allocation: `torch.empty()` on the existing activation memory pool.

**Option B — module attribute (REJECTED)**
- Storing on `self` breaks autograd — the cache tensor would outlive the forward call and confuse the saved-tensor lifetime tracking.
- Would also not work under TP/PP (different ranks share the module but have different activations).

**Option C — global dict keyed by hash (REJECTED)**
- Hash of `v` and `psi` would require reading every element, defeating the purpose.
- Even with weakref identity, gets broken by CUDA graph replays (see §2 caveat).

### Chosen pattern

```python
# Forward (autograd.Function.forward)
psi_v_cache = torch.empty(B, S, H, R, P, dtype=V.dtype, device=V.device)
mamba_mimo_fwd(..., V, MIMO_V, ..., PsiV_out=psi_v_cache)
ctx.save_for_backward(Q, K, V, ..., psi_v_cache)

# Backward (autograd.Function.backward)
Q, K, V, ..., psi_v_cache = ctx.saved_tensors
mamba_mimo_bwd_fwd(..., V, MIMO_V, ..., PsiV_in=psi_v_cache)
# (bwd_bwd similarly)
```

---

## 5. Invalidation triggers

N/A for intra-step cache.  The tensor lives exactly from `forward` to `bwd_bwd` of the same autograd op and is released by autograd when the graph is freed.

Cross-step invalidation is not needed because we do not cache across steps.

---

## 6. Memory overhead estimate

For NAM56R single layer (B=1 micro-batch, S=8192, H=16, R=4, P=64, bf16 = 2 B):

```
sizeof(PsiV) = 1 * 8192 * 16 * 4 * 64 * 2 bytes = 64 MiB per sample per layer
```

Per micro-batch at MBS=8: **512 MiB / layer × 9 DSA + 4 full = 13 Mamba3 layers per model stack**
Actually NAM56R uses Mamba3 mixers at **52 layers total** per `project_nam56r_attention_layout.md`. Reviewing `plan.md§50-53`:

> Memory: 5.6 GiB/rank for cache (9 layers × 0.62 GiB)

Production-plan estimate: **~5.6 GiB/rank for the full model**. That's within budget (current peak ~132 GiB at MBS=8, +5.6 GiB = ~4% of remaining headroom).

If the number turns out tighter (e.g. MBS=10 push), we can:
- Quantize PsiV cache to fp8 on write, fp16 on read (bwd is bf16 anyway — noisy but maybe acceptable)
- Save only at chunk boundaries and recompute within-chunk — saves `R×` factor
- Only cache for layers where bwd_bwd is the bottleneck (selective)

---

## 7. Expected speedup

From `plan.md§50`:
> `Mamba3 P2 PsiV cache (5 days, +1.5-2.3% = 274-277 TFLOP/s on bench3)`

From `docs/mamba3_mimo_p3_register_split_design.md` line 118-119, hoisting PsiV was called out as *"free if we cache post-rotary Q/K + PsiV per Task B of GB10 P2 investigation"*.

### Dissection of the 1.5-2.3% estimate

**Per-kernel PsiV work** (baseline from `reference_mamba_ssm_optimization_plan.md`):
- `mamba_mimo_bwd_fwd`: 1034 ms / step on H200 — PsiV recompute is ~5% of that = **52 ms**
- `mamba_mimo_bwd_bwd`: 2110 ms / step — PsiV recompute is ~3% of that + **register-pressure relief** from dropping ~3 frag tiles. Conservative savings 100-200 ms.

Total direct savings: **~150-250 ms per step**. At 5540 ms / step iteration time, that's **2.7-4.5% kernel-level**, translating to roughly **1.5-2.3% end-to-end TFLOP/s** (because Mamba3 mixers are not the only kernel in the step). Matches plan estimate.

### Independent verification

No existing profile data directly measures "kernel minus PsiV recompute". The estimate is derived from kernel decomposition, not A/B measurement. **The real speedup must be measured on H200 after implementation**, exactly as P1 was measured selectively (`docs/mamba3_mimo_p1_notes.md` addendum).

### Risk: speedup may be lower than 1.5%

If the TileLang compiler is already CSE-ing `psi_v = v * psi` across its internal scheduling stages (e.g. hoisting the load and keeping the product in a register between back-to-back `ct.mma` calls), then the runtime cost per PsiV recompute is near-zero and we get nothing. Possible on well-scheduled WGMMA code. We will **measure nsys before committing real work**.

---

## 8. Files created (this session)

1. **`docs/mamba3_mimo_p2_psiv_cache_design.md`** (this file) — design.
2. **`cppmega/megatron/mamba3_psiv_cache.py`** — Python module skeleton (TODO stubs only).
3. **`cppmega/megatron/upstream_patches/apply_mamba3_p2_psiv_patches.py`** — patch-applier skeleton, mirrors `apply_mamba3_mimo_p1_patches.py` structure. **Not wired into any shim yet.**
4. **`tests/test_mamba3_psiv_cache.py`** — correctness-test scaffolding. Skips by default (no GPU / no patch applied).

Nothing in this series modifies production code paths. Env gate `CPPMEGA_MAMBA3_P2_PSIV_CACHE` is read and, if "1", raises `NotImplementedError` — we will never silently pretend the cache is on.

---

## 9. Integration sketch (future sessions)

### Phase A — Python wrapper (no kernel changes yet, baseline for perf)

Drop-in subclass of `cppmega_tilelang_mimo_combined` that computes `psi_v` at the Python level before calling the kernel, passes it through as if it were V (materialising the product), and compares correctness. This tells us the *ceiling* of the technique without any TileLang work. If even this hack doesn't move the needle, abandon.

### Phase B — fwd kernel outputs PsiV

Add a new output tensor to `mamba_mimo_fwd`: an extra argument + a `ct.store(PsiV_out, psi_v, index=...)` inside the loop. Autograd wrapper saves it.

### Phase C — bwd_fwd / bwd_bwd read PsiV instead of recomputing

Replace

```python
v = ct.load(V, ...)
v_bc = ct.broadcast_to(v.reshape((CHUNK, 1, P)), (CHUNK, R, P))
psi_bc = ct.broadcast_to(psi_bf.reshape((1, R, P)), (CHUNK, R, P))
psi_v = v_bc * psi_bc
```

with

```python
psi_v = ct.load(PsiV_IN, index=(i_b, c, i_h, 0, 0), shape=(1, CHUNK, 1, R, P)).reshape((CHUNK, R, P))
```

Dropping ~3 fragment tiles from the inner live set.

### Phase D — perf measurement + gate flip

Nsys on bench3 at NAM56R shape. If ≥2% total TFLOP/s gain and no regression on bwd smem, flip `CPPMEGA_MAMBA3_P2_PSIV_CACHE` default ON.

Correctness criterion (matching P1 / TMA layout fix): `rel_err < 0.02` on all 14 gradient tensors vs the pre-patch baseline.

---

## 10. Upstream patch vs fork

We have not decided yet. Two options:

1. **Upstream-style patch** (mirrors P1): in-place edit the installed `mamba_ssm/ops/tilelang/mamba3/*.py` kernels at import time. Pros: composable with P1. Cons: adding kernel signature arguments requires also patching `mamba3_mimo.py` (the Python autograd op). Multi-file, brittle.

2. **Fork into cppmega** (mirrors tilelang_mimo_autograd): new kernel file + new autograd Function `cppmega_tilelang_mimo_p2_combined`. Selected via env gate. Lives entirely in cppmega. Pros: no upstream file touching, independent testing. Cons: diverges from upstream, double kernel-compile cost at import.

**Recommendation for implementation**: **Option 2** for initial correctness work (safer), convert to Option 1 once stable and once we're confident we want to propose upstream.

---

## 11. Blocker check (pre-flight)

Before investing kernel-work time, confirm these **are not** blockers:

- [ ] **Autograd saved-tensor overhead**: Adding a ~5.6 GiB tensor to `save_for_backward` — does PyTorch clone it? (Should be a view; need to confirm on bench3.)
- [ ] **CUDA graphs compatibility**: The cache tensor must be allocated inside the graph-captured region. If `torch.empty()` at each call breaks graph capture, we need a pool allocator. Easy to retrofit.
- [ ] **TileLang signature bloat**: `mamba3_mimo_fwd` already has 18 kernel arguments. TileLang may have a soft limit — needs a one-liner smoke test.
- [ ] **Per-rank activation memory**: NAM56R peak memory is ~132 GiB at MBS=8 on H200-141. +5.6 GiB is fine. But at MBS=10 (bench3 new record 269.4 TFLOP/s, ref `reference_main_head_liger_ce_gap.md`), budget is tighter. Check.

---

## 12. Status summary

| Item | Status |
|---|---|
| Design doc | **done** (this file) |
| `mamba3_psiv_cache.py` skeleton | **done** (TODO stubs) |
| `apply_mamba3_p2_psiv_patches.py` skeleton | **done** (TODO stubs) |
| `tests/test_mamba3_psiv_cache.py` scaffolding | **done** (skip-unless-GPU + skip-unless-patch-applied) |
| Python-level prototype (Phase A) | not started |
| Kernel changes (Phases B/C) | not started |
| Perf measurement | not started |
| Integration into smoke script | not started |
| Gate default | **OFF** (will stay OFF until verified) |

## 13. Next session TODO

1. **Phase A Python prototype** (1 day): materialise `psi_v = v * psi` in the Python wrapper before calling the kernel, feed as V. Measure nsys delta. Gate the whole P2 pursuit on this result: if no measurable perf change, abandon.
2. **If Phase A wins**: start Phase B (fwd kernel PsiV output). On bench3 H200 (not GB10 — GB10 smem cap blocks `bwd_bwd` at NAM56R shape per P1 notes). ~2 days.
3. **Phase C** (bwd kernel PsiV input): most of the real work, ~2 days.
4. **Perf + correctness validation**: nsys before/after, rel_err check on all 14 grads, compare iter-25 loss to baseline (should be BF16-noise-identical).
5. **Doc + commit flip**: env-gate default OFF → ON only after H200 perf confirms ≥1.5%.

## 14. If Phase A shows no gain — archive

If the Python-level materialisation shows no measurable TFLOP/s improvement (within 0.5% noise), then the TileLang compiler is already handling PsiV optimally and this work should be **archived**. Update this doc with a "superseded / not pursued" addendum and point future agents at the Phase A measurement as the reason.
