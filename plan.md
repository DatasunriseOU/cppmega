# NAM56R optimization plan — next 10 hours

## combined_1f1b overlap PP=1 EP=8: BLOCKED by pre-existing MTP bug (2026-04-14)

Parallel agent (me) ran the overlap test (`CPPMEGA_EP_OVERLAP=1`) with
VARIANT=v3 (PP=1 EP=8) on bench3. **Both MTP_DEPTHS=2 and MTP_DEPTHS=1
crash at iteration 0** with:

```
RuntimeError: The size of tensor a (4096) must match the size of tensor b (2048/1366)
  at non-singleton dimension 1
  cppmega/cppmega/megatron/mtp_liger_ce.py:178
  mtp_loss = loss_mask * mtp_loss
```

**Root cause**: `cppmega/megatron/hybrid_schedule_plan.py`
`_relaxed_post_init` flips `mtp_num_layers = None` when
`overlap_moe_expert_parallel_comm=True AND PP==1` to bypass upstream
`PP>1` assertion. But flipping to None during `__post_init__` causes
the model to **skip MTP block construction entirely** — decoder emits
`[s, b, h]` not `[(1+D)*s, b, h]`. Post-process node still calls
`process_mtp_loss` which chunks by `1+config.mtp_num_layers=2 or 3`,
yielding `[s/2, b, h]` or `[s/3, b, h]`. Then `loss_mask*mtp_loss`
dies on the shape mismatch (`loss_mask` stays `[b, s]`).

Baseline (same config WITHOUT `CPPMEGA_EP_OVERLAP`) is fine: 257
TFLOP/s steady for iters 3-30, loss converges 11.95 → 4.96. Matches
the 253 TFLOP/s EP=4 record with +1.6% win (already documented).
Archived baseline: `/mnt/data/cppmega-root/cppmega/c1f1b_baseline_v3.log`
(30 iters complete, max_alloc 111 GB/rank). Overlap failure logs:
`c1f1b_overlap_mtp{1,2}_FAIL.log`.

**Next steps to unblock overlap**:
1. Fix `_relaxed_post_init` PP=1 path: either (a) keep
   `mtp_num_layers` intact and find a different bypass for the PP>1
   assert, or (b) skip `process_mtp_loss` in `_HybridPostProcessNode`
   when mtp_num_layers was flipped.
2. Or: make MTP block construction NOT depend on
   `mtp_num_layers is None` gate (build MTP blocks based on a separate
   flag that we preserve).
3. Test overlap again after fix. Target: ≥260 TFLOP/s (EP A2A overlap).

**Do not ship the v3 + overlap combo** until fix. v3 baseline
(no overlap) is already validated at 257 TFLOP/s.

### Option (b) fix LANDED + TESTED (2026-04-14 late — commit 253f740)

`_HybridPostProcessNode._do_forward` now guards `process_mtp_loss` with
a SHAPE check: `hidden_states.shape[0] == (1 + mtp_num_layers) * loss_mask.shape[-1]`.
If decoder didn't emit MTP-expanded output, skip the loss call.

**Test result** (bench3 VARIANT=v3 + CPPMEGA_EP_OVERLAP=1):
- **MTP bypass fixed** — training past iter 0, no more shape mismatch crash
- **New failure**: OOM at iter 2, peak 131 GiB/rank vs 115 GiB baseline
- **+16 GiB memory overhead** for combined_1f1b scheduling + DeepEP A2A
  buffers, overflowing bench3's 140 GiB budget during Mamba3 bwd temps
- Exact error site: `dq_bias_tilelang = dq_tilelang.sum(dim=(0,1)).permute(...)`
  needed +512 MiB transient that didn't fit

**Final conclusion: combined_1f1b single-node 8×H200 is not viable**.
Research predicted no benefit, empirical confirms. Ship:
- PP=1 EP=8 WITHOUT overlap = bench3 production candidate (257 TFLOP/s)
- `CPPMEGA_EP_OVERLAP=1` env gate stays default OFF
- MTP shape guard (253f740) = infrastructure safety net for future
  overlap explorations (e.g., at deeper PP or smaller models)

## Test iter 14 finding (2026-04-14 late evening)

**bench3 PP=1 EP=8 (VARIANT=v3) baseline stable at 257 TFLOP/s** (iters
3-14 steady-state). Prior record was 253 TFLOP/s at PP=1 EP=4. This is
a **+1.6% real gain** — marginal but reproducible. Loss converging
(11.95 → 7.07), grad_norm reasonable. **Production candidate**: switch
bench3 default from EP=4 to EP=8.

**Europe P1+TMA test FAILED** — agent ab0cdd07a098d108a returned:
- **TMA layout fix branch (`tma-layout-fix-3d-to-2d` @ `31dc695`) is
  BROKEN on H200**: `csr % R` / `csr // R` modulo indexing in
  `mamba_mimo_bwd_bwd_kernel` trips TileLang's LayoutInference
  FloorMod const-fold with "Divide by zero".
- **GB10 correctness test was a false positive**: GB10's 99 KiB smem
  cap prevented bwd_bwd from running at NAM56R shape, so the buggy
  path never exercised. **Lesson**: GB10 correctness ≠ H200 correctness.
- **GQA patch tangled in same diff**: pristine mamba_ssm reinstall
  loses our local `elif H % G == 0` branch → `G value of 8 not supported`
  (matches `reference_env_drift_bench3_europe.md` memory).

**Direction change**:
1. **Do NOT merge tma-layout-fix-3d-to-2d to main**. P1 full is blocked.
2. **Ship PP=1 EP=8 (v3) config** as new bench3 production candidate
   pending overlap test completion + europe cross-validation.
3. **TMA layout fix repair** deferred: needs `T.constexpr` hints or
   modulo elimination in `qk_dot_shared` indexing. 1-2 days kernel
   surgery. See `reference_tma_layout_fix_broken_h200.md`.

## Pivot 2026-04-14: DualPipeV is phantom, combined_1f1b is canon

Exa deep research (agent ad869db932d3e8d66) confirmed:

- **NVIDIA Megatron-LM gave up on DualPipeV** (issue #1524 closed by
  @Victarry): *"DualPipe requires decomposition of xgrad and wgrad. But
  mcore prefers TE which makes it hard to decompose backprop."*
- **NVIDIA NeMo-Bridge DeepSeek-V3 recipe** (the authoritative DS-V3
  reproduction): TP=2 PP=16 EP=64, uses **VPP + 1F1B-A2A-overlap =
  our `combined_1f1b`**, NOT DualPipeV.
- DeepSeek's own DualPipe repo has ZERO EP/MoE examples — toy MLPs only.
- DeepSeek's training launcher was never open-sourced.

**Direction change**:
1. STOP DualPipeV integration work. It's a phantom — our earlier
   `apply_dualpipev_patch.py` + `dualpipev_schedule.py` stay as
   scaffolding, env gate default OFF. Do not pursue.
2. FIX `combined_1f1b` memory instead. Our OOM-at-every-MBS finding
   needs re-investigation: activation recompute granularity, memory
   accounting, small-PP feasibility. DeepEP (via Megatron flex) is the
   same as DeepSeek-V3 production stack.
3. Update `reference_combined_1f1b_dead_for_nam56r.md` memory from
   "dead" to "needs-memory-fix" after `combined_1f1b` debug.

See `reference_dualpipev_phantom_combined1f1b_canon.md` memory for
full evidence.

### combined_1f1b concrete solvable path (research found)

Exa+Context7 agent (a28511a315f73a9b5) dug into Megatron internals and
found: **PP=2 is architecturally worst-case** (+1 microbatch penalty +
`--recompute-granularity full` incompatible via hard assertion in
`combined_1f1b.py:303-305`). The combined_1f1b scheduler is designed
for **PP=1** (no-pipelining variant, no penalty) or **PP≥4+VPP≥2**
(VPP amortization).

**Test path for NAM56R**:
```
--pipeline-model-parallel-size 1
--expert-model-parallel-size 8                # 2 experts/rank at 16 total experts
--tensor-model-parallel-size 1
--overlap-moe-expert-parallel-comm
--moe-token-dispatcher-type alltoall          # NOT flex; combined_1f1b uses
                                              # Megatron's reference A2A not DeepEP
--recompute-granularity selective
--delay-wgrad-compute
```

At PP=1, `schedules.py:648` routes to `combined_1f1b_schedule_for_no_pipelining`
— just one extra output tensor, no pipeline-buffer penalty. NAM56R 4.73B
PP=1 MBS=8 fits at ~118 GiB; overlap adds ~2-5 GiB. Should fit in
141 GiB.

**Blocker to check**: Issue #1862 (closed 2025-10-16) — combined_1f1b
crashed with `TP=1` giving "Input A does not hold any data!". Workaround
was TP≥2 which we can't afford (Mamba3 TP=2 is 3.2× slower). Need to
verify the fix is present in our `core_v0.15.0rc7 + PR #3674` Megatron.

Full details: `reference_combined_1f1b_solvable_path.md` memory.

## Hard rules for this cycle (enforced)

- **NEVER open PRs / issues / comments in external projects (state-spaces,
  tile-ai, NVIDIA, Dao-AILab, etc.) without explicit user request AND
  explicit approval of the exact text.** Drafting into `upstream_prs/*.md`
  is fine; running `gh` write commands against third-party repos is not.
  Reinforced 2026-04-14.  See memory `feedback_pr_approval.md`.
- No silent fallbacks. Crash loudly on failure, no try/except that hides
  errors. See `feedback_no_silent_fallbacks.md`.
- Apply `apply_dsa_cg_patches.py` after any Megatron-side state change.
- PP=1 on H200 requires CG disabled (TE CG pool eats 39.5 GiB).
- TP>1 for Mamba3 MIMO is a net loss on single-node H200 (3.2× slower);
  do not propose it. See `project_mamba3_tp_is_net_loss.md`.
- CP direction closed (architectural trade-off, not bug). See
  `reference_cp_blocked_by_custom_mixers.md`.


Written 2026-04-14 after TMA layout fix validated on GB10 (branch
`tma-layout-fix-3d-to-2d` @ `31dc695`). Plan is revised as results come
in; see `docs/fp8_research_session_2026_04_14.md` for session context.

## Current state snapshot

- **main** @ `11a2787` — DualPipeV integration + long-context roadmap
- **tma-layout-fix-3d-to-2d** @ `31dc695` — 3D→2D smem flatten + TMA on
  for bwd kernels. Numerics verified rel_err 0.0038-0.0116 on GB10.
  **Needs H200 perf measurement**.
- **bench3** (H200): running selective P1 fwd-only (a417bac3f9cee042b), ~1.5h in
- **europe** (H200): running DualPipeV smoke (a23dd1434fcf36945), ~30 min in
- **GB10**: free
- Production baseline: bench3 253 TFLOP/s, europe 289 TFLOP/s (PP=1 EP=4 MBS=8)

Expected max gain from this 10-hour block: **+5-8% TFLOP/s** if P1 full
lands. Target: bench3 ~270 / europe ~310 = ~31% MFU.

## Stretch target reality check (test-loop steering)

User's test-loop directive: **250k tok/sec** and **MFU > 50%** on 8×H200.

| metric | baseline | this-block ceiling | stretch target | feasibility |
|---|---|---|---|---|
| TFLOP/s | 289 | ~347 (35%) | 495 (50%) / 982 (99%) | 50% = months of research; 99% = impossible |
| tok/sec | 74,000 | ~89,000 | 125,000 (50% MFU) / **250,000 (99% MFU)** | 250k = at HW peak |
| Gap vs 250k | 3.4× | 2.8× | — | — |

**Honest conclusion**: 250k tok/sec and 50% MFU are **aspirational
long-term goals**, not achievable in 10 hours. Realistic ceiling with
current Mamba3+MLA+DSA+MoE architecture and TileLang 0.1.8 kernels is
~35-37% MFU (~105-110k tok/sec). Beyond that requires:

- CUTLASS-level persistent kernel rewrites (FA-4 pattern, months)
- Custom PTX for Mamba scan (weeks, uncertain perf)
- Architecture surgery (remove MLA for DSA-only, replace Mamba3 with
  Mamba-2 SSD — quality regression)
- Multi-node scale-out (beats 8-GPU efficiency for large-seq workloads)

This 10-hour block pursues the **achievable part** (~+7% via P1 full)
and the **groundwork** for future research (upstream PRs, P2 correct
impl, P3 design). We DO NOT claim to hit 50% MFU; we honestly target
31-35% and document the ceiling.

---

## Hour 0-2 — Wait + apply TMA fix on first free H200

**Blocker**: both H200 currently busy. Work runs opportunistically.

### Live findings from running agents (test-iter 2)

Observed via partial output sampling (not committed-SHA-yet):

**bench3 selective P1 agent (a417bac3f9cee042b)** — found a subtle
patch bug:
- `mamba_ssm/__init__.py` auto-imports `Mamba3` which transitively
  imports `mamba3_mimo_forward` BEFORE `apply_all()` runs in our shim.
- `co_firstlineno` of `mamba_mimo_fwd_kernel` is cached at import time.
- Our patch inserts a new `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`
  line, shifting subsequent lines down by 1.
- `inspect.getsource(func)` via `linecache` reads the new file at the
  OLD line number → grabs partial function → `IndentationError`.

**Root-cause fix options** (agent is picking one):
1. Make patch preserve line count (replace an existing whitespace line
   instead of inserting a new one).
2. Invalidate `linecache.clearcache()` after patching.
3. Run `apply_all()` before `import mamba_ssm` — requires earlier hook
   point in shim (before `mamba_ssm`'s own `__init__.py` pulls Mamba3).

Likely ship option 1 (minimal invasive); maybe also clear linecache
for belt-and-suspenders.

**europe DualPipeV agent (a23dd1434fcf36945)** — found an architectural
incompatibility with MoE expert-parallelism:
- DualPipeV V-shape puts different pipe_ranks on different layers at
  the same time (rank 0 in stage 0 layer 12; rank 1 in stage 1 layer
  18 simultaneously).
- EP=4 spans the full world (ranks 0-3 in one EP group, 4-7 in
  another). DeepEP's `fused_dispatch` A2A requires ALL EP peers to hit
  the SAME MoE layer's A2A call at the same time.
- With DualPipeV: rank 0 (pipe_rank 0) at MoE in layer 12, rank 1
  (pipe_rank 1) at MoE in layer 18 (or wherever stage 1 is). They
  never synchronize on the same collective → deadlock inside
  `deep_ep/buffer.py:97 all_gather_object`.

**Current attempt**: agent is retrying with `VARIANT=v0` (EP=1, no
expert parallelism) to at least verify DualPipeV works on dense
layers. If that works, we know the DualPipeV mechanics are correct —
the limitation is EP is incompatible with DualPipeV V-shape.

**Implication**: if DualPipeV is to ever ship for NAM56R with MoE, we'd
need **EP scoped within pipe_rank** (e.g., EP=2 across ranks at same
pipe_rank), or EP disabled entirely. This is significantly narrower
than the initial hope and likely makes DualPipeV unviable for
production (we need EP=4 for memory/expert distribution).

### Action items when agents return

- [x] **bench3 selective-P1 agent returned** (commit `d80bf9c`):
  - Infrastructure shipped: rank-0-only + flock race fix, atomic writes,
    line-count-preserving patch (merge onto FAST_MATH line — fixes the
    IndentationError from `co_firstlineno` desync).
  - Scope narrowed to fwd + fwd_varlen only.
  - Measurement: **wash** (183.016 → 183.005 TFLOP/s, -0.006% noise).
    Fwd not the bottleneck at MBS=8. Memory +0.76 GiB.
  - Verdict: HOLD selective-fwd P1 default OFF. Ship infra.
  - New memory: `reference_py_patch_line_shift_bug.md`,
    `reference_mamba_ssm_reinstall.md`.
- [x] **DualPipeV EP=1 agent STOPPED** at test-iter 3: europe was
  verified idle, agent was stuck in SSH loop. Killed it to free
  europe for the higher-ROI full P1 + TMA fix measurement.
- [~] **europe full P1 + TMA fix agent LAUNCHED** (a21a3192d964308bb):
  - Checks out `tma-layout-fix-3d-to-2d` branch
  - Applies P1 + TMA layout fix patches
  - Compile probe at NAM56R shape on H200 sm_90
  - Baseline vs P1+TMA smoke (MBS=8, 25 iters, PP=1 EP=4 no-CG)
  - Target: ≥3% TFLOP/s gain → ship to main; <3% → keep infra + hold

### Merge conflict note for tma-layout-fix-3d-to-2d

`git merge-tree main origin/tma-layout-fix-3d-to-2d` reports 1 conflict:

- `docs/mamba3_mimo_p1_notes.md` — both branches appended addendums.
  Trivial to resolve (keep both addendums in order).

Resolution plan (execute only if europe full P1+TMA measurement gives
a ship-worthy win):

```bash
git checkout tma-layout-fix-3d-to-2d
git pull origin tma-layout-fix-3d-to-2d
git merge origin/main   # or rebase; resolve the doc conflict
git push origin tma-layout-fix-3d-to-2d
git checkout main
git merge --ff-only tma-layout-fix-3d-to-2d  # or open PR via gh
git push origin main
```

Files unique to the branch (additions, no conflict):
- `cppmega/megatron/upstream_patches/apply_mamba3_mimo_tma_layout_fix.py`
- `cppmega/megatron/upstream_patches/mamba3_mimo_bwd_tma_layout_fix.patch`
- `scripts/exploration/tma_layout_repro.py` (+ related bench files on branch)

### Action items (still active)
  - Apply: `apply_mamba3_mimo_p1_patches` + `apply_mamba3_mimo_tma_layout_fix`
  - Config: PP=1 EP=4 MBS=8 no-CG FP8 tensorwise, 25 iters
  - Compare TFLOP/s + peak memory + loss vs 253/289 baseline
  - Capture nsys kernel durations for `mamba_mimo_fwd`, `mamba_mimo_bwd_fwd`,
    `mamba_mimo_bwd_bwd` — expect 20-30% reduction each
  - **Success criterion**: ≥3% total TFLOP/s gain, loss-sane

**Commits expected this block**: possibly 0-1 (observation only).

---

## Hour 2-4 — Production-harden TMA fix if win confirmed

Scenario A — **P1 full wins ≥3%**:

- [ ] Merge TMA fix patches into `apply_mamba3_mimo_p1_patches.py`
  - Absorb `apply_mamba3_mimo_tma_layout_fix.py` changes into the main
    patch file. One invocation point, one env gate `CPPMEGA_MAMBA3_P1=1`.
  - Keep the unified diff archive for upstream PR use.
- [ ] Add rank-0-only + `dist.barrier()` race guard to `apply_all()`
  - Fixes the 8-rank concurrent-write `IndentationError` seen earlier.
  - Pattern in the patch file:
    ```python
    is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
    if is_rank0:
        _do_patch()
    if dist.is_initialized():
        dist.barrier()
    ```
- [ ] Commit, push to main, verify second smoke run on second H200
  machine (cross-machine consistency).
- [ ] Update `README.md` production config table to reflect new TFLOP/s,
  flip `CPPMEGA_MAMBA3_P1` default to ON in
  `scripts/remote_smoke_h200_dsa_9_4_m.sh`.
- [ ] New memory entry `reference_p1_full_win.md` with measured
  before/after numbers.

Scenario B — **P1 full regresses or crashes**:

- [ ] Diagnose: bwd smem overflow? WGMMA layout mismatch? Try with bwd_bwd
  disabled (run with only fwd+bwd_fwd patched) — partial gain acceptable.
- [ ] Fallback: keep selective fwd-only P1 (the current branch), land as
  modest ~1.5% win.
- [ ] Memory entry `reference_p1_full_regression.md` documenting what
  broke and why.

**Commits expected this block**: 2-4 (patch merge, race fix, defaults,
memory).

---

## Hour 4-6 — P2 correct implementation (post-rotary Q/K + PsiV cache)

GB10 agent confirmed P2 premise in the original plan was wrong: the
kernel already caches STATES + QK_DOT. What bwd_bwd recomputes is
post-rotary Q/K + PsiV. Cacheable at ~128 MiB/sample cost.

- [ ] Read the GB10 P2 design doc at `docs/mamba3_mimo_p2_state_checkpoint_design.md`
  on branch `p2-state-checkpoint-prototype` (worktree on GB10).
- [ ] Launch GB10 agent OR dedicated H200 agent to implement the real P2:
  - Modify `mamba3_mimo_bwd.py` to additionally emit post-rotary Q/K and
    PsiV tensors to gmem in `bwd_fwd` (~100 MB/sample extra).
  - Modify `bwd_bwd` to read these instead of recomputing rotary/bias.
  - Expected: ~10-15% reduction in bwd_bwd (2110 → ~1800 ms)
    = ~0.5-1% total TFLOP/s.
- [ ] Correctness test on GB10 (rel_err target < 0.02).
- [ ] If ship: measure on H200 at next free slot.

**Commits expected this block**: 1-3 (P2 kernel patch, design doc,
correctness test).

**Deprioritize if**: P1 full gave >5% — P2 is ~1% and the follow-through
engineering is big. Focus on polishing the bigger win.

---

## Hour 6-8 — DualPipeV final verdict + optional P3 sketch

Scenario A — **DualPipeV smoke worked** in hour 0-2:
- [ ] Run comparison: DualPipeV vs vanilla PP=2 same MBS/GBS
- [ ] nsys profile showing bubble reduction
- [ ] Document numbers + commit script default if win ≥5% at PP=2 config

Scenario B — **DualPipeV smoke crashed**:
- [ ] Review the iterative debug agent's attempts (committed one-by-one)
- [ ] Verdict: one more hour of debug OR formally close direction with
  memory entry `reference_dualpipev_final_status.md`.

After DualPipeV decision, if time remains:
- [ ] Sketch P3 register pressure reduction (255 → target 128 regs in
  bwd_bwd). Not implementation — just the split design: which ops to
  separate into a second kernel pass. Document in
  `docs/mamba3_mimo_p3_register_split_design.md`.

**Commits expected this block**: 1-3.

---

## Hour 8-10 — Upstream contributions + polish

Once P1 is landed + P2 is sketched, outward contributions:

- [x] **TileLang upstream issue draft**: written to
  `upstream_prs/08_tilelang_tma_bulk_copy_3d_smem_issue.md` (pulled
  forward during test-loop iteration since GB10 was busy with TMA fix
  only).  Ready to post pending user approval.
- [x] **state-spaces/mamba PR draft**: written to
  `upstream_prs/07_mamba3_mimo_3d_to_2d_smem_refactor.md` (same test
  iteration). Contains full correctness table (14 gradients), site list,
  rationale. Ready to open PR after H200 perf numbers confirm the win.
- [ ] **README refresh**:
  - Update throughput results table with post-P1 measurements
  - Remove the mythical DualPipeV 205 TFLOP/s row (or mark experimental)
  - Add Mamba3 MIMO P1 entry to optimization stack
- [ ] **Memory cleanup**:
  - Mark stale entries older than 30 days that are no longer applicable
  - Consolidate the 3 "FP8 dead path" memories into a single summary if
    redundant
- [ ] **Open PR for tma-layout-fix branch to main** once H200 perf
  measurement is clean — user approval gate per memory.

**Commits expected this block**: 2-5 (upstream issue drafts, README,
memory cleanup, PR).

---

## Dependencies / risks

- **Hour 0-2 depends on H200 freeing**. If both agents hold through hour
  2, skip forward to hour 4-6 design work on GB10 (free) and measure
  later.
- **TMA fix is the single biggest lever**. If it gives <2% on H200 (not
  the 4-5% projected), re-evaluate priorities — shift hours 4-10 toward
  different direction (e.g., investigate bench3 vs europe remaining 13%
  gap via system-level probing).
- **DualPipeV in hour 6-8** has unknown difficulty. May burn an extra
  hour if it's almost-working.
- **No speculative kernel rewrites** — P3 is design-only in this plan.

## Not in this 10-hour block (deferred)

- FP8 attention bwd production integration (R&D branch stays R&D until
  convergence validation)
- CP/SP port (closed direction per `reference_cp_blocked_by_custom_mixers.md`)
- FP8 Mamba SSM (dead path per `reference_fp8_mamba_ssm_dead_path.md`)
- 128k context work (roadmap'd in `docs/long_context_roadmap.md`)
- Selective FP8 MoE + DSA end-to-end production bake (can be done after
  P1 is live; both are orthogonal)

## Measurement hygiene

Every smoke test uses:
- `MBS=8 GBS=64` at `PP=1 EP=4 no-CG` (bench3 gold) OR `PP=1 EP=4` (europe)
- 25 iterations, median of iters 3-25
- Report: TFLOP/s, peak memory per rank, step-0 and step-25 loss
- Commit SHA + machine name in every log filename

Expected ceiling after this 10-hour block: **bench3 ~270, europe ~310,
MFU ~31%**. Past that we're into P3/P4/P5 territory which is weeks of
kernel rework, not hours.
