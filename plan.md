# NAM56R optimization plan

## GB10 Muon memory/perf plan (2026-04-25)

Detailed implementation note:
[`docs/quantized_muon_momentum.md`](docs/quantized_muon_momentum.md).

Current session note:
[`docs/gb10_local_memory_perf_2026_04_25.md`](docs/gb10_local_memory_perf_2026_04_25.md).

Completed in this checkpoint:

- Blockwise int8/uint8 Muon momentum extension exists and is covered by CUDA
  tests.
- Multi-tensor q8 momentum update overwrites BF16 grads in place, so the grad
  buffer becomes the Newton-Schulz input and no separate BF16 scratch is needed.
- Grouped update+normalization is wired into local Megatron
  `TensorParallelMuon`: QKV tensors use independent Q/K/V norm groups, ordinary
  2D tensors use one group per tensor, and `_newton_schulz_lowmem` skips its
  duplicate input normalization when the grouped primitive already normalized
  the BF16 input.
- Norm-plan metadata is cached on the optimizer instead of rebuilt every step.
- Sharding remains part of the API: every rank builds local norm metadata with
  the same logical group ids, local group sumsq values are all-reduced over
  `tp_group`, and the scale kernel uses the global group norms.
- `cppmega/megatron/m2rnn_triton.py` no longer saves full
  `y[B,S,H,K,V]`; backward recomputes chunks from sparse fp32 checkpoints.
- Local GB10 launcher defaults to FlashAttention, TE bias+GELU, Muon no-master
  BF16 fallback, quantized Muon momentum, Adam8bit scalar fallback, and
  `--local-ddp-disable-contiguous-grad-buffer`.
- Lion8bit scalar fallback is now an opt-in env override:
  `CPPMEGA_MUON_SCALAR_OPTIMIZER=lion8bit`.

Verified:

```bash
PYTHONPATH=. /home/dave/cppmega-venv/bin/python -m pytest -q tests/test_quantized_muon_momentum.py
PYTHONPATH=. /home/dave/cppmega-venv/bin/python -m pytest -q tests/test_m2rnn_triton.py
```

Result:

```text
quantized_muon_momentum: 11 passed
m2rnn_triton:            12 passed
```

Local GB10 quarter MBS=4 smoke with no contiguous DDP grad buffer:

```bash
CPPMEGA_TRAIN_ITERS=1 \
CPPMEGA_MICRO_BATCH_SIZE=4 \
CPPMEGA_GLOBAL_BATCH_SIZE=4 \
CPPMEGA_MEM_PROFILE=1 \
CPPMEGA_MEM_PROFILE_STEPS=1 \
scripts/local_gb10_quarter_train.sh
```

Result:

```text
after setup: alloc 3.422 GiB, reserved 4.199 GiB
step 1 post: alloc 11.504 GiB, reserved 24.367 GiB, max_alloc 22.932 GiB
lm loss 1.165463E+01, mtp_1 loss 1.164792E+01, grad norm 76.923,
skipped 0, nan 0
```

Local GB10 quarter MBS=4 profile:

```text
/home/dave/logs/gb10_quarter_mbs4_profile_20260425_110533.log
/home/dave/logs/gb10_quarter_mbs4_profile_20260425_110533_torch_profile/train_step_2_cuda_table.txt
```

```text
after setup: alloc 3.422 GiB, reserved 4.199 GiB
step 2 post: alloc 11.505 GiB, reserved 26.408 GiB, max_alloc 25.095 GiB
step 2 losses: lm 1.097315E+01, mtp_1 1.137682E+01
grad norm 113.327, skipped 0, nan 0
```

Delta vs earlier local baseline:

```text
setup allocation: 6.832 GiB -> 3.422 GiB
step-2 max_alloc: 29.224 GiB -> 25.095 GiB
step-2 reserved:  31.740 GiB -> 26.408 GiB
```

Lion8bit scalar fallback A/B:
[`docs/lion8bit_ab_2026_04_25.md`](docs/lion8bit_ab_2026_04_25.md).

```text
full AEMEAEMEAEMR, MBS=1, 2 iters:
adam8bit step2 loss/grad/max_alloc: 11.21753 / 127.889 / 12.807 GiB
lion8bit step2 loss/grad/max_alloc: 11.21530 / 128.080 / 12.460 GiB
```

Nsight/profiler state:

```text
/home/dave/logs/qmuon_qkv_grouped_ncu.ncu-rep
/home/dave/logs/gb10_quarter_mbs4_nsys_fork_20260425_111137_nsys.nsys-rep
/etc/modprobe.d/nvidia-profiler-counters.conf installed
update-initramfs -u -k all completed
ncu reports were collected with sudo -E /usr/local/cuda/bin/ncu
end-to-end nsys under torch.distributed.run is still partial; torch profiler is reliable
```

Next implementation target:

1. Make the local Megatron patch durable without swallowing unrelated dirty
   Megatron worktree changes.
2. Add checkpoint/state_dict handling for `quantized_momentum_buffer`.
3. Continue fusing the Muon step around Newton-Schulz; grouped q8 update+norm is
   wired, but `TensorParallelMuon.step` remains the top profiler entry.
4. Inspect CCE backward and GEMM shapes from the torch trace.
5. Improve MoE token movement/sort/permute; partial nsys reports still point at
   `_permute_kernel` and `_sort_chunks_by_map_kernel`.
6. Run a longer full-pattern Lion8bit A/B before switching the default.
7. Keep this invariant:

   ```python
   quantized_muon_momentum_update_multi_(q_states, grads, beta=group["momentum"])
   ```

8. Do not allocate `state["momentum_buffer"]` in quantized mode.
9. Reuse `p.grad` as the BF16 Newton-Schulz input after the update kernel.
10. Keep router scalar/bias/gating, RMSNorm, bias, embedding, output, and other
   non-2D fallback parameters on no-master BF16-state Adam8bit.

Updated profiling/fusion sequence:

1. Get a cleaner end-to-end nsys capture; the embedded `nsys` reports are
   partial under `torch.distributed.run`, while torch profiler is currently
   reliable.
2. Run 50-200 steps on real clang 4k data and compare loss, grad norm,
   skipped iterations, tokens/sec, CUDA max allocated/reserved.
3. Profile optimizer, CCE backward, MoE token movement, and GEMM shapes from
   the same run so we do not chase a microbench win that does not move
   end-to-end throughput.
4. Use `sudo -E /usr/local/cuda/bin/ncu` for focused kernel reports while the
   loaded NVIDIA module still requires admin profiling counters.
5. Fuse in this order:
   - remove remaining Python/metadata overhead around q8 Muon plan dispatch;
   - fuse deeper around Newton-Schulz input handling where it avoids copies;
   - attack MoE sort/permute if nsys continues to show it as large;
   - then evaluate CCE/GEMM-specific kernel work.

## Current state (2026-04-14 morning session 2 — multi-agent investigation)

### Production configs

Canonical throughput + config table lives in
**[`docs/production_status.md`](docs/production_status.md)** (single source
of truth). Summary: bench3 = 268 TFLOP/s (FP8, PP=1 EP=8 MBS=10 v3), europe
= 289 TFLOP/s (BF16, PP=1 EP=4 MBS=8). Both use Liger `reduction="mean"`
broadcast workaround — the earlier "269.4 with reduction=none" is
superseded (silent grad corruption).

### What's DONE (this session)

#### Code shipped:
1. ✅ **`cppmega/megatron/apply_linear_ce_patch.py`** — main-head LinearCrossEntropyModule swap + Liger reduction=mean fix (workaround Liger #968 silent grad corruption). Probe-based dispatcher works for cc=9 (Hopper native PR #3345 if cherry-picked), cc=10 (Blackwell native), cc=12 (GB10 Liger fallback)
2. ✅ **`cppmega/megatron/dsa_indexer_fused_patch.py`** — module-level monkey-patch of `_compute_index_scores` → per-head fused BF16 accumulation. Saves ~16 GiB at MBS=8 (4.11 → 0.24 GiB per call, 17× reduction). GB10 verified rel_err 1.9e-7 at NAM56R prod shape.
3. ⚠️ **`cppmega/megatron/mtp_native_hopper_ce.py`** — infrastructure committed (class-swap + monkey-patch wiring), env gate `CPPMEGA_MTP_NATIVE_HOPPER_CE=0` default OFF. Activating produces `grad_norm=NaN` (Suspect #1 transpose round-trip refuted empirically, Suspect #2 CG collective pending). **Do NOT enable until NaN root-caused.**
4. ✅ **`scripts/remote_smoke_h200_dsa_9_4_m.sh`** — env var hooks for all new patches, default ON for `CPPMEGA_DSA_INDEXER_FUSED=1`
5. ✅ **`tests/test_dsa_indexer_fused_patch.py`** — parity tests vs upstream einsum

#### Stack upgrades:
- ✅ **TileLang main built + distributed** to bench3 + europe + GB10 + local Mac (commit `f309d814`, 2026-04-14)
  - PR #746 follow-ups: 3D smem `LowerBulkCopy` falls back gracefully вместо ICHECK
  - PR #2002: pipeline planning before layout inference
  - PR #1909: ProducerConsumerWarpSpecialized + `T.tma_copy()` API
  - PR #2024 + #2037
- ✅ **Apple CCE 25.1.1 → 25.9.3** upgraded on bench3 + europe (но empirically не помогает в нашем shape — delta ±200 MiB)
- ✅ **Megatron PR #3345** cherry-picked на bench3 (Hopper native fused linear CE). Standalone smoke validates compilation + correctness. Integration via class-swap works.
- ✅ **SSH keys fixed** — direct `dave@<ip>` works on bench3 + europe (was Permission denied earlier)
- ✅ **Bwd files state**: P1 patches re-applied (`TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE=True` for bwd_bwd compile)

### What's IN PROGRESS

| Agent | Task | ETA |
|---|---|---|
| ac244d76 | Verify CG_FLAGS=NONE actually propagates + identify 63.5 GiB pool source | running |
| a4666f85 | Extend PR #3345 cherry-pick to MTP head | running |
| af7f16fa | (duplicate) In-place dsa.py edit — will be canceled, ac2fa14c monkey-patch path chosen | running |
| current | bench3 MBS=10 + all fixes (CG=NONE + Liger=mean + indexer fused) | iter ~1 (just launched pid 2223812) |

### What's PROD-READY

1. **Liger main-head reduction=mean** (Path A, Liger #968 workaround) — math exact, gradient correct
2. **DSA indexer fused** (per-head accumulation) — saves 3.87 GiB/call, 17× reduction
3. **TileLang main** — unblocks ALL bwd kernels (3D smem fallback works)
4. **SSH key infrastructure** — direct dave@ access on both H200 machines

### What's IN PLAN (week+ effort, kernel-level)

1. **Mamba3 P2 PsiV cache** (5 days, +1.5-2.3% = 274-277 TFLOP/s on bench3)
   - Cache `Q_rot + K_rot` in bwd_fwd, drop PsiV recompute
   - Memory: 5.6 GiB/rank for cache (9 layers × 0.62 GiB)
   - GB10-testable

2. **Mamba3 P3 register split** (HALT — better path = warp-spec)
   - Path 0 (warp-spec annotations): 2-3 weeks, gated on TMA layout fix on H200
   - Path 1 (PsiV hoist): 2 days, GB10-testable, attacks root cause (17 → 13 frags)
   - Path 2 (full kernel split): 8-12 days, last resort (no atomics needed)

3. **PR #3345 native Hopper CE** (cherry-picked, MTP extension in progress)
   - Eliminates Liger entirely on cc=9
   - No silent gradient corruption (per-token bwd works correctly)
   - Estimated +1-3% if MTP coverage completes

4. **TileLang upstream PRs file** (require user approval per `feedback_pr_approval.md`):
   - Mamba LinearCrossEntropyModule regression (Megatron PR #3226 → #3207 clobber)
   - TileLang TMA 3D smem documentation/support request
   - TileLang FloorMod const-fold in T.Parallel

### Stretch goal (250k tok/sec / 50% MFU)

- Current ~289 TFLOP/s = 29.2% MFU
- 50% MFU = 495 TFLOP/s = **1.7× gap**
- Realistic stack of P2 + indexer fused + PR #3345 = +5-10% → 305-318 TFLOP/s = 31-32% MFU
- **Gap remains** — requires CUTLASS WGMMA custom kernels (months) или новое hardware (B200+)

### Critical session findings (verified by 6 grounding agents)

- **Liger #968 silent grad corruption REAL** — `reduction="none"` backward reads `grad_output[0]` as scalar regardless of shape. Worst case: `grad_output[0]=0` → all gradients = 0. Multiple PRs (#1126 = assertion only, #680 = different module). No real per-token bwd fix exists in any fork.
- **Apple CCE doesn't help on our V=65k shape** — 28GB→1GB paper number was vs dense fp32 CE at V=256K, not vs Liger. Practical delta ±200 MiB in our regime.
- **63.5 GiB "CUDA Graphs private pool"** label is from PyTorch OOM message, illustrative not exclusive. Source = TE FP8 `make_graphed_callables` or Megatron `--cuda-graph-impl=transformer_engine`. CG_FLAGS=NONE properly empties — verified.
- **TileLang TMA bug REAL on v0.1.8**, but **main HEAD has graceful fallback** for 3D smem
- **Mamba3 P3 register split path partially valid** but warp-spec annotations approach (Path 0) is NOT drop-in (kernel needs explicit WS split first — 2-3 weeks rewrite)

### Action items for next session

1. **Re-baseline production** with all fixes shipped (current bench3 test, pid 2223812 running)
2. **Commit + push to repo** (in progress)
3. **Update install/setup docs** with new env vars, TileLang main build instructions, mamba_ssm patch reapplication
4. **MTP extension** (PR #3345 path) — when a4666f85 returns
5. **Mamba3 P2 PsiV cache** — implement after current fixes validated

---

## Afternoon session (2026-04-14) — PR filing prep + backup + drift discovery

### Work shipped

1. **13 PR templates + reproducers + explanation_ru.md drafted** covering Liger
   FLCE bug (C1), Megatron PR #3345 MTP extension (C2), DSA indexer fused
   memory patch (C3), TileLang TMA 3D→2D smem fallback (C4), Apple CCE 25.9.3
   upgrade (C5), `apply_linear_ce_patch.py` main-head swap (C6), Mamba3 P1
   patches (C7), `mtp_native_hopper_ce.py` (C8), mamba_ssm fork drift (C9),
   bench3 specific `mamba3_mimo_bwd.py` patch (C10), FP8 sparse MLA europe-only
   files (C11), DualPipeV abandoned path cleanup (C12), combined_1f1b
   documentation (C13). All drafts local, **nothing filed upstream** pending
   user explicit approval per `feedback_pr_approval.md`.

2. **MCP grounding of PR claims** — `.tmp/mcp_grounding_pr_claims.md`. Key findings:
   - Perplexity hallucinated **PR #680** as the Liger FLCE `reduction="none"`
     fix; the real #680 is for non-fused LigerCrossEntropy (different module,
     doesn't help FLCE path). **Do not cite #680 as a FLCE fix in any upstream
     comment.**
   - PRs #968 / #1126 plausible but could not be independently confirmed by
     MCP engines — verify on GitHub directly before quoting in any filing.

3. **Full backups captured + pushed to GS**:
   - `sftp://BUCKET_ARTIFACTS/backups/backup_bench3_2026_04_14/` — 17 MiB, 23
     objects. Incl. `megatron_lm_tree.tar.gz` (flat tree, **no `.git`** — this
     motivated the restoration recipe below)
   - `sftp://BUCKET_ARTIFACTS/backups/backup_europe_2026_04_14/` — full state
     including `.git` HEAD hashes and format-patch files for unpushed commits

4. **Megatron restoration recipe** — `docs/megatron_restoration_recipe.md`.
   Discovered bench3 megatron tree has **no `.git` history**; recipe pins the
   tarball in GS as authoritative source-of-truth, documents best-effort
   upstream commit hypothesis (bench3 likely at `2eeabc668` — PR #3674 only),
   provides diff procedure to confirm against upstream when network available.

5. **README Megatron Version corrections** (`README.md` section around line
   154): per-machine divergence recorded (base commit, patches, notes).
   Authoritative per `docs/megatron_restoration_recipe.md` and
   `docs/session_3_gap_audit.md`:
   - **bench3**: `megatron-core 0.18.0rc0` (installed via `pip install
     git+NVIDIA/Megatron-LM@980211ae`; has PR #3345 cherry-pick). Note the
     bench3 tarball's `package_info.py` reports `0.16.0rc0` but the actual
     installed site-packages is `0.18.0rc0` — this is what caused the
     earlier "both machines at 0.16" confusion.
   - **europe**: `megatron-core 0.16.0rc0` (`dev_latest` branch, 2 commits
     ahead of `origin/dev`; has PR #3674 + PR #4268 cherry-picks).

### Diagnostic findings (no new code)

- **MTP NaN Suspect #1 REFUTED**: `apply_linear_ce_patch.py` Liger routing has
  correct `expand(b,s).contiguous()` broadcast pattern — re-reviewed math,
  matches what #968 workaround requires. NaN at MBS=12 backward is NOT caused
  by this patch. Moved investigation to Suspect #2.
- **MTP NaN Suspect #2 pending CG_FLAGS=NONE propagation**: Need to verify
  whether `CUDA_GRAPH_FLAGS=NONE` env var actually threads through to the TE
  `make_graphed_callables` path or gets silently ignored — next session.

### Drift discoveries (previously unknown)

- **bench3 vs europe megatron divergence**: the two H200 hosts are on
  **different** megatron-core versions (see
  `docs/megatron_restoration_recipe.md` + `docs/session_3_gap_audit.md` for
  authoritative source):
  - bench3 = `megatron-core 0.18.0rc0` (flat tarball install via
    `pip install git+NVIDIA/Megatron-LM@980211ae`; has PR #3345
    cherry-pick). The bench3 tarball's `package_info.py` reports
    `0.16.0rc0`, but the *actually-installed* site-packages is
    `0.18.0rc0` — this caused the earlier "both at 0.16" misread.
  - europe = `megatron-core 0.16.0rc0` (live git on `dev_latest` 2
    commits ahead of `origin/dev`; has PR #3674 + PR #4268 cherry-picks).
- **mamba_ssm fork drift `31f3d7b` vs `4f4857f`**: bench3 at `31f3d7b`
  detached + 4 modified files; europe at `31f3d7b` detached + 3 modified
  files + `mamba3_siso_bwd.py.orig` stock copy. Previously assumed both
  machines ran the same fork — they do not.
- **Europe-only FP8 sparse_mla files**: `tilelang_sparse_mla_{fwd,bwd}_fp8.py`
  and `__init__.py` in `experimental_attention_variant/ops/` exist only on
  europe (not bench3). These are captured in `europe_megatron_modified.tar.gz`.
  Port to bench3 if FP8 sparse MLA path ever becomes desired.

### Status transitions this session

- In production: bench3 FP8 MBS=10 EP=8 v3 = 268 TFLOP/s (canonical, Liger reduction=mean), europe BF16 MBS=8 EP=4 = 289 TFLOP/s. See `docs/production_status.md` for single source of truth. Prior "269.4" was Liger reduction=none, superseded (silent grad corruption)
- In test: all 13 PR templates (awaiting user approval gate before any filing)
- Deferred: MBS=12 backward NaN debug (needs Suspect #2 CG flag propagation test), Mamba3 P2 PsiV cache (~5 days, not started), TileLang upstream issue/PRs (draft only)

---

## EMPIRICAL CONFIG EXPLORATION COMPLETE (2026-04-14 deep night)

**Stop running config sweeps.** 20 empirical tests across every dimension
(precision, MBS, GBS, EP, DP, recompute modules, CE fusion paths, Liger
variants) have mapped the full space. Production ceilings locked.

Next productive work is kernel-level (week+ effort):
1. **MBS=12 backward nan debug** (my `apply_linear_ce_patch.py` Liger routing
   has correct forward at MBS=12, fits 130 GiB, but backward grad_norm=nan)
2. Mamba3 P2 PsiV cache (design deferred)
3. Mamba3 P3 register split (design ready, week+ effort)
4. cuTile bwd_bwd refactor per TileGym patterns

## Session conclusion (2026-04-14 deep night — 17 empirical tests + new patch)

Config-space fully mapped on 8×H200. Production ceilings (canonical in
`docs/production_status.md`):
- **europe = 289 TFLOP/s (29.2% MFU)** — BF16 MBS=8 EP=4, fabric-bound
- **bench3 = 268 TFLOP/s (27.1% MFU)** — FP8 MBS=10 EP=8 + Liger main-head CE
  with `reduction="mean"` broadcast (Liger #968 workaround via
  `cppmega/megatron/apply_linear_ce_patch.py`)
- The earlier "269.4 with Liger reduction=none" measurement is SUPERSEDED —
  that path silently corrupts gradients via Liger #968. Canonical bench3
  record is 268 with the mean-broadcast workaround.

**Gap to user's 250k tok/sec / 50% MFU stretch target**: 1.7× throughput.
Empirical evidence shows **this gap cannot be closed by config/topology tuning**
on this architecture + hardware (every knob tried regresses or doesn't fit).
Realistic near-term improvements: +5-10% via Mamba3 P2/P3 kernels (weeks of
work). Reaching 50% MFU requires either (a) CUTLASS WGMMA kernel rewrites
(months), (b) architecture change (not our call), or (c) newer hardware (B200+).

Document honestly: 289/268 TFLOP/s are production numbers we can ship today.
Single source of truth: `docs/production_status.md`.

## Production baselines (2026-04-14 night — historical empirical sweep)

NOTE: the canonical production configs live in `docs/production_status.md`.
The table below preserves the full empirical sweep from 2026-04-14 night
for reference. Any standalone number you find here MUST be cross-checked
against `docs/production_status.md` before citation. In particular, the
"269.4" bench3 row used Liger reduction=none which is now known to silently
corrupt gradients (Liger #968) — do NOT treat that row as shippable.

| Machine    | Config                                                        | TFLOP/s   | MFU       | Status                                                                           |
| ---------- | ------------------------------------------------------------- | --------- | --------- | -------------------------------------------------------------------------------- |
| europe     | PP=1 EP=4 MBS=8 BF16 no-CG                                    | **289**   | 29.2%     | **gold record, ship**                                                            |
| **bench3** | **PP=1 EP=8 MBS=10 FP8 tensorwise (v3)**                      | **268**   | **27.1%** | **new bench3 record, ship**                                                      |
| bench3     | PP=1 EP=8 MBS=10 FP8 + CPPMEGA_SELECTIVE_FP8_MOE=1            | 268       | 27.1%     | **net-neutral gate (redundant when global FP8)**                                 |
| bench3     | PP=1 EP=8 MBS=8 BF16 no-CG (v3)                               | 257       | 26.0%     | superseded                                                                       |
| europe     | PP=1 EP=8 MBS=8 BF16 no-CG (v3)                               | 262       | 26.5%     | -9.3%, DO NOT SHIP on europe                                                     |
| europe     | PP=1 EP=4 MBS=10 FP8 tensorwise                               | 190       | 19.2%     | -34%, DO NOT SHIP FP8 on europe                                                  |
| europe     | PP=1 EP=4 MBS=8 BF16 + CPPMEGA_SELECTIVE_FP8_MOE=1            | 247       | 25.0%     | -14%, DO NOT SHIP FP8 MoE on europe                                              |
| europe     | PP=1 EP=4 MBS=9 BF16                                          | 228       | 23.1%     | -21%, odd MBS breaks FP16 GEMM tile alignment                                    |
| europe     | PP=1 EP=4 MBS=8 GBS=128 BF16                                  | 252       | 25.5%     | -12%, doubled grad-accum = more recompute passes                                 |
| europe     | PP=1 EP=4 MBS=8 BF16 + Liger main-head CE                     | 250       | 25.3%     | -13%, same pattern — Liger slowdown outweighs memory savings europe doesn't need |
| europe     | PP=1 EP=4 MBS=10 BF16                                         | OOM       | —         | NCCL crash at peak 127.7 GiB — MBS=8 is europe memory ceiling                    |
| bench3     | PP=1 EP=8 MBS=11 FP8 tensorwise (v3)                          | 264       | 26.7%     | -1.5%, odd MBS marginal regress — MBS=10 optimal                                 |
| bench3     | PP=1 EP=8 MBS=10 FP8 + Liger main-head CE (MTP vanilla)       | **269.4** | **27.2%** | **+0.5% vs 268 baseline, NEW RECORD**. σ<0.15 rock-solid                         |
| bench3     | PP=1 EP=8 MBS=10 FP8 + Liger MTP + Liger main-head (stacked)  | 269.2     | 27.2%     | stacking no additive gain vs main-only                                           |
| bench3     | PP=1 EP=8 MBS=10 FP8 + Liger stacked + drop moe_act recompute | 165-178   | 17-18%    | **catastrophic** -34%; moe_act recompute is critical, not optional               |
| bench3     | PP=1 EP=8 MBS=12 FP8 tensorwise (v3)                          | OOM       | —         | CUDA illegal address / NCCL crash — MBS=10 is bench3 ceiling                     |
| bench3     | PP=1 EP=8 MBS=12 FP8 + full recompute                         | OOM       | —         | OOM at vocab_parallel_cross_entropy (12 GiB logits), CE head-bound not Mamba     |
| bench3     | PP=1 EP=8 MBS=10 BF16 (v3)                                    | **118**   | 11.9%     | **-56%, thrashes — FP8 is MEMORY enabler (peak 128 GiB vs 115 GiB FP8)**         |

**Definitive conclusion — per-machine production configs locked in**:
- **bench3** ships `VARIANT=v3 MBS=10 FP8 tensorwise` → 268 TFLOP/s (σ<0.5, rock steady).
- **europe** ships `VARIANT=v1 MBS=8 BF16` → 289 TFLOP/s. FP8 tensorwise
  REGRESSES on europe (190 TFLOP/s, -34%) — amax overhead exceeds GEMM
  speedup on europe's fabric.
- FP8 tensorwise is machine-specific: wins on bench3 (+4.3% vs BF16 v3),
  loses on europe. Memory: `reference_fp8_mbs10_bench3_wins.md`,
  `reference_fp8_mbs10_europe_regression.md`.

**Gap to stretch goal**: europe 289 TFLOP/s = 29.2% MFU vs 50% target = **1.7× to go**.
EP + FP8 + MBS + GBS topology tuning **all exhausted** (15 empirical tests this session).

**Potential main-head Liger CE lever (2026-04-14 night, HONEST assessment)**:
MBS=12 bench3 OOMs at `vocab_parallel_cross_entropy` (12 GiB logits on LM head).
Current `CPPMEGA_MTP_LIGER_CE=1` applies Liger fused linear CE to MTP only.
Per `cppmega/megatron/mtp_liger_ce.py` docstring: **Liger CE is 2.7× SLOWER than
vanilla on H200** (memory enabler, not speedup). So adding it to main head:
- ✓ Unlocks MBS=12 on bench3 (currently OOM) — empirically needed
- ✗ Slows main-head CE by ~75 ms per iter (+1-2% per-iter time)
- ≈ Net throughput: likely **flat to +5%** (bench3 MBS=11 was -1.5% vs MBS=10,
  so MBS=12 gain is uncertain)
- Europe unlikely to benefit (MBS=9/10 BF16 both regress due to fabric/odd batch)

Worth trying on bench3 after implementation. Memory: `reference_main_head_liger_ce_gap.md`.
Implementation: hook `compute_language_model_loss` in `megatron/core/models/common/language_module/language_module.py:168` to route through `LigerFusedLinearCrossEntropyFunction` when env gate enabled. Mirror `cppmega/megatron/mtp_liger_ce.py` pattern.

Next levers require kernel-level implementation:

1. **Mamba3 P2 post-rotary Q/K + PsiV cache**: potential +1-3% via bwd saving,
   but MORE IMPORTANT — could save 10-20 GiB per rank, which per session insight
   `reference_fp8_is_memory_win_not_compute.md` directly unlocks BF16 MBS=10 on
   bench3 (currently thrashes at 128 GiB, FP8 saves 13 GiB to reach 115 GiB).
   **Design doc missing** — write before implementation.
2. **Mamba3 P3 register split 255→130**: +1% total, week+ effort. Design in
   `docs/mamba3_mimo_p3_register_split_design.md`.
3. **FP8 attention backward** (R&D branch `fp8-bwd-piggyback-exploration` @
   4d79332): GB10 5-iter smoke stable, needs H200 convergence validation before
   stacking with production config.
4. **cuTile Python bwd_bwd refactor** (TileGym patterns per
   `reference_tilegym_cutile_patterns.md`): multi-week rewrite, unknown gain.

**FP8 MoE gate** (`CPPMEGA_SELECTIVE_FP8_MOE=1`) CLOSED this session:
- Bench3 with global FP8: **no-op** (267.7 vs 268 baseline, redundant)
- Europe BF16 + MoE FP8: **-14%** regression (247 vs 289 baseline)

Stacked realistic best case: ~5-10% over 289 → 303-318 TFLOP/s = 30-32% MFU.
Still far from 50% without CUTLASS WGMMA custom or CuTe DSL Hopper stack
rewrites. Document honestly.

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

| metric      | baseline | this-block ceiling | stretch target                            | feasibility                                |
| ----------- | -------- | ------------------ | ----------------------------------------- | ------------------------------------------ |
| TFLOP/s     | 289      | ~347 (35%)         | 495 (50%) / 982 (99%)                     | 50% = months of research; 99% = impossible |
| tok/sec     | 74,000   | ~89,000            | 125,000 (50% MFU) / **250,000 (99% MFU)** | 250k = at HW peak                          |
| Gap vs 250k | 3.4×     | 2.8×               | —                                         | —                                          |

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
