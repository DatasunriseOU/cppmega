# GB10 mamba3 bwd_bwd optimization — final conclusion (2026-04-11)

After exhaustive attempts across three paths, **TileLang 167 µs is the ceiling for `mamba3_mimo_bwd_bwd` on GB10 (sm_121a)**. We ship TileLang. Further optimization effort on GB10 mamba3 kernels is not productive; focus moves to H200 where WGMMA / TMA / swizzled smem unlock different wins not available on consumer Blackwell.

## Update 2026-04-11 — Modal B200:2 variant re-sweep

**Key new data point:** the same 5 cuTile variants that regressed on GB10 were re-tested on **Modal B200:2 (sm_100a datacenter Blackwell, TMEM present, 228 KiB smem)**. Two of them **flipped from losers to winners**:

| Variant | B200 µs (−% vs baseline) | GB10 µs (−% vs baseline) |
|---|---|---|
| Baseline 2-kernel A/B | 687.6 (1.00×) | 624 (1.00×) |
| **V3 3-kernel split** | **460.5 (−33%) WINS** | 678 (+9% regressed) |
| V4 hoisted invariants | 622.3 (−10%) wins | 742 (+19% regressed) |

**Same algorithmic variant, opposite outcome on two Blackwell chips.** Reason: the launch-overhead vs live-set-savings trade-off depends on the smem budget per kernel. B200's 228 KiB rewards splitting into 3 kernels; GB10's 99 KiB dynamic budget punishes the extra launch overhead with no compensating register relief.

**Structural gap persists on B200:** even the v3_split3 winner is **2.57× slower than TileLang** (460 µs vs 179 µs bwd_bwd) on B200 despite full hardware advantages (TMEM, tcgen05, 2.3× smem, 22.5× TFLOPS). The cuTile gap is compiler-model structural, not a hardware limit. See `docs/modal_b200_cutile_variant_sweep_2026_04_11.md` for the full sweep.

**Also new:** `@ct.kernel(occupancy=1)` is a silent no-op on cuTile 1.2.0 on both GB10 and B200 — accepted without error, zero measurable effect. Abandoned as a tuning knob.

**Also new:** cuTile `mamba3_mimo_fwd` is **17.7% FASTER than TileLang on B200** (0.054 ms vs 0.064 ms per the 2026-04-10 parity run). This is a deployment opportunity — a hybrid `cuTile fwd + TileLang bwd` wrapper is a pure throughput win on B200 deployments.

**Revised recommendations:**

1. **GB10 default**: keep the 2-kernel A/B split baseline (624 µs). V3 3-kernel split regresses on GB10.
2. **B200 default** (if cppmega ever ships a cuTile bwd for B200): ship `variant_v3_split3.py` at 460 µs. 33% faster than baseline but still 2.57× slower than TileLang.
3. **Universal default**: **stay on TileLang for production bwd_bwd** on any Blackwell. The 2.5-3× cuTile gap is structural.
4. **Easy B200 win**: hybrid `cuTile fwd + TileLang bwd` wrapper for B200 deployments (+17.7% on fwd-dominated workloads).
5. **Cross-HW rule**: never assume a cuTile algorithmic variant that wins on one GPU will win on another — always re-sweep on the target HW.

## Three paths tried, same conclusion

### Path 1 — cuTile Python algorithm rewrite

Five variants tried (see `.tmp/cutile_bwd_bwd_rewrite/RESULTS.md`). **All regressed** vs the 2-kernel baseline (624 µs).

| Variant | µs | vs baseline |
|---|---|---|
| **Baseline 2-kernel A/B split** | **624** (optimal) | 1.00× |
| Nested `@ct.function` per phase | 1498 | 2.40× slower |
| Fused monolithic | 1405 | 2.26× slower |
| 3-kernel split | 678 | 1.09× slower |
| Hoisted loop invariants | 742 | 1.19× slower |
| `ct.static_iter` unroll | 3236 | 5.20× slower |

**Verdict:** The existing 2-kernel split is pareto-optimal. Kernel fission via gmem temps is the only reliable live-range cut in cuTile Python — fusing regresses, nested helpers regress, hoisting regresses, full unroll catastrophically regresses. See `memory/reference_cutile_compiler_behavior.md` for the full list of compiler behavior findings.

### Path 2 — CuTe DSL BF16 hot-path hybrid

Empirically verified CuTe DSL BF16 warp MMA + TMA + persistent scheduler work on sm_121a with zero patches via `cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py`. Ported the hottest GEMM (`lkq = k_trap @ q_rot^T` batched across L=256 chunks).

**Results:**
- CuTe DSL batched GEMM L=256: **10.28 µs** (bit-exact vs torch.einsum)
- **`torch.bmm((L=256, 64, 64), ...) = 10.33 µs`** — cuBLAS on GB10 already matches hand-written CuTe DSL kernel at 64×64×64 BF16
- Pure-torch reference of cuTile kernel A chunk loop: **12,306 µs** (20× slower than cuTile — Python dispatch overhead)
- CUDA-graph lower bound for all batched GEMMs in a host-level hybrid: **~280-400 µs** (unreachable floor)

**Verdict:** Host-level hybrid (CuTe DSL or torch.bmm) cannot beat cuTile's 624 µs let alone TileLang's 167 µs. The 4× TileLang-vs-cuTile gap is **kernel structure**, not GEMM efficiency. TileLang fuses all 10 GEMMs + ~150 elementwise ops + rotary + reductions into **one CUDA kernel with 16 CTAs each running 16 chunks in on-chip state** (registers + smem, no DRAM round-trip). cuTile Python cannot replicate that model; it must split into at least 2 kernels with gmem temps.

### Path 3 — Triton M²RNN autotune sweep (europe H200)

Not directly for bwd_bwd, but relevant because M²RNN R-layers are part of the same NAM56R stack.

Full 25-config space on 3 shapes (B=2/4, H=8/16, K=64, V=16). **Current defaults already optimal** — Triton autotuner had already converged to `num_warps=8`. `num_stages` is irrelevant (<0.03% delta) because M²RNN is a sequential recurrence holding state in registers — nothing to pipeline.

**Verdict:** No end-to-end win; 0% change. Triton already picks the optimal config on production shape.

## Actionable outputs

### Ship / keep
- **TileLang baseline** (`mamba_mimo_bwd_combined` at 167 µs on GB10 with `TL_ENABLE_AGGRESSIVE_SHARED_MEMORY_MERGE`) — production path
- **cuTile 2-kernel A/B split** at 624 µs — kept as a Python-readable reference for future algorithmic changes, not a production path
- **Triton M²RNN kernel at `num_warps=8`** — current defaults are correct, no change needed

### Report upstream
- **NVIDIA CUTLASS `examples/79_blackwell_geforce_gemm/dense_gemm.py` bug**: `tile_shape_mnk=(64,64,64)` gives wrong results on sm_121a at L≥180 (0.3% of output cells wrong with abs diff ~60 when expected ~2). Workaround: `tile_shape_mnk=(128,64,64)` or larger. Reproduced with `dense_gemm_smoke.py --mnkl 64,64,64,180 --tile_shape_mnk 64,64,64`. Worth a GitHub issue to NVIDIA CUTLASS.

### Optional cleanup (not shipped without user approval)
- Prune `cppmega/megatron/m2rnn_triton.py` autotune grid: drop `num_warps ∈ {1, 2}` (never picked, 2.2× and 1.7× slower), add `num_stages=5` for completeness. Saves ~40% cold-sweep time at zero steady-state cost.

### Untried algorithmic lead (future work)
Replace `(FUSED, FUSED)` masked GEMMs on the diagonal paths (`dqk_from_diag`, `dk_intra`, `dq intrachunk`) with per-chunk `(R, R)` batched GEMMs → **16× FLOP reduction on the diagonal path**. Blocked by a cuTile 1.2.0 `TileSyntaxError` on generator-based `ct.cat` — need to pre-materialize the batched view as a precomputed `Array.slice` or use explicit per-element addressing. This is the biggest remaining potential win for cuTile bwd_bwd if someone can work around the syntax error.

## Where optimization effort should go instead

GB10 mamba3 bwd_bwd is pareto-optimal at TileLang 167 µs. Move effort to:

1. **H200 mamba3 bwd_bwd** — WGMMA + TMA + swizzled smem + warp specialization via `setmaxnreg` are all available there. CuTe DSL hybrid (CuTe-hot + cuTile-simple) is a live optimization path on sm_90a. See `docs/nam56r_mimo7_baseline_2026_04_11.md` next-steps section.
2. **NAM56R MIMO 7/7 end-to-end** — the full stack runs at 56,280 tok/sec baseline with 4.44× gap to the 250k target. CUDA graphs, FP8 selective, nsys profile, swizzle-layout fix are all much higher-leverage than a 3× cuTile bwd_bwd improvement would be.
3. **`fp8_path_status.md` Path C re-test under FP8** — now that TileLang ngroups=8 bwd is unblocked and MIMO 7/7 trains cleanly in BF16, FP8 on the non-scan layers is the next throughput lever on bench3 H200.

## References

- `memory/reference_cutile_compiler_behavior.md` — the 5 compiler behavior findings
- `.tmp/cutile_bwd_bwd_rewrite/RESULTS.md` — full per-variant data
- `/home/dave/mamba3_mimo_cute_dsl/` on GB10 — all CuTe DSL port artifacts
- `/home/dave/mamba3_mimo_cutile/variants/` on GB10 — all cuTile rewrite artifacts
- `/tmp/m2rnn_sweep_report.json` on europe — Triton autotune sweep data
- `docs/nam56r_mimo7_baseline_2026_04_11.md` — the full NAM56R MIMO 7/7 baseline and next-steps
- `docs/gb10_sm121_hardware.md` — what sm_121a actually supports
- `docs/gb10_software_stack.md` — working stack recipe for GB10
