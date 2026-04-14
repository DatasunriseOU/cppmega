# NAM56R optimization plan — next 10 hours

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

- [ ] When bench3 selective-P1 agent returns: review result
  - If +1% neutral: the selective fwd-only patch lands, but TMA fix will
    make it obsolete in hour 2
  - If OOM/crash: debug the race-condition fix
- [ ] When europe DualPipeV smoke returns: review result
  - If works: good, reserve europe for full P1 measurement later
  - If blocks: document blocker, close DualPipeV direction OR schedule
    deeper debug in hour 6-8 block
- [ ] Immediately when a H200 frees: **launch full P1 + TMA fix smoke**:
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
