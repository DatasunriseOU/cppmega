# GB10 software stack recipe — 2026-04

What libraries, versions, compile flags, and workarounds actually work on `GB10 DGX Spark` (sm_121a, consumer Blackwell) today. Complement to `docs/gb10_sm121_hardware.md` — that covers the silicon, this covers the software. Sourced from NVIDIA CUTLASS / TensorRT-LLM / FlashAttention / Triton / FlashInfer / vLLM issue trackers, NVIDIA DevTalk threads, and direct empirical verification on real GB10 hardware (2026-04-11).

## What to use for each goal

| Goal on GB10 | Working path | Avoid | Perf / notes |
|---|---|---|---|
| **BF16 / FP16 mamba3 / attention kernels** | TileLang baseline **OR** CuTe DSL warp MMA + TMA + persistent scheduler (newly verified working on sm_121a) **OR** cuTile Python 1.2.0 | tcgen05 paths (silicon-blocked), FP4 without the `a`-suffix patch | TileLang 167 µs bwd_bwd is the known reference; CuTe DSL BF16 path empirically proved end-to-end 2026-04-11 via `cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` |
| **NVFP4 dense GEMM** | CUTLASS example 79 compiled with `-arch=sm_120f` | `-arch=sm_121a` (9× slower), B200 tile defaults that overflow the 99 KiB smem budget | **356 TFLOPS** (71% of peak sparse) per NVIDIA forum benchmark |
| **NVFP4 GEMM via TRT-LLM** | TRT-LLM ≥ 2026-04-09 (CUTLASS submodule bumped to 4.4.2 by @depaulmillz), or `nvfp4_gemm_cublaslt` path on earlier versions | `nvfp4_gemm_cutlass` on TRT-LLM < 1.3.0rc2 with CUTLASS < 4.4.2 — silent numerical corruption | CUTLASS 4.4.2 landed `StageCountAutoCarveout` fix for issue #3144 |
| **NVFP4 MoE grouped GEMM** | CUTLASS 4.4.2 with 256×128 or 128×128 tile shape | Default SM100 tile 128×256×256 — overflows 99 KiB smem | 120-154 TFLOPS measured |
| **NVFP4 in vLLM** | Env vars: `VLLM_USE_FLASHINFER_MOE_FP4=0`, `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1` | FlashInfer FP4 MoE — crashes during CUDA graph capture | 42 → 50 tok/s, -7 GB memory |
| **Attention (general purpose)** | **PyTorch SDPA** (efficient-attention backend via cuDNN) | Building FA2 / FA3 from source for expected "speedup" — same perf as SDPA (489 ms vs 481 ms on InternVL3-8B/S=512) | FA2/FA3 give NO speedup on GB10 |
| **Attention (hand-rolled for max perf)** | **CUTLASS PR #3030** (`blake-snc` / Second Nature Computing) — BF16 CpAsync + TMA + FP8 inline-PTX, benchmarked on real GB10 | FlashAttention-4 (physically impossible, tcgen05/TMEM silicon absent) | ~42 TFLOPS FP8 validated; PR unmerged as of 2026-04 but usable |
| **CuTe DSL BF16/FP16 warp MMA + TMA** | **WORKS OUT OF BOX on sm_121a** via `cute.nvgpu.warp.MmaF16BF16Op` + `cute.nvgpu.cpasync.CopyBulkTensorTile{G2S,S2G}Op` + persistent scheduler. Use `cutlass/examples/python/CuTeDSL/blackwell_geforce/dense_gemm.py` as reference. Pass `"sm_120"` explicitly as SmemAllocator capacity key | `CUTE_DSL_ARCH=sm_100a` / `sm_120a` overrides (cubin runtime-reject), `sm_121` without the `a` suffix (NVVM compile error) | Empirically verified 2026-04-11: ~38 µs/iter at 1024³ persistent BF16 GEMM on real GB10 |
| **CuTe DSL tcgen05 or FP4/NVF4/MXF4 ops** | BLOCKED: tcgen05 is silicon-absent (no fix possible). FP4/NVF4 warp-level is a **30-line allowlist monkeypatch** away from working (`MmaSM120BlockScaledOp.admissible_archs`) | — | `cutlass/cute/arch/numeric_conversion.py:319` already whitelists sm_121a for related ops — the `MmaSM120BlockScaledOp` omission appears to be a miss, not an intentional block |
| **Triton custom kernels** | Upstream Triton main + `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` | Triton PR #9734 revision (broken `tensormap.replace`, reverted in #9755); scatter4-using kernels | Perf ≈ sm_80 baseline; MoE / MXFP4 has gaps |
| **MXFP8 / NVFP4 GEMM via cuBLAS** | cuBLAS 13.2 — release notes explicitly say "Improved GEMM performance on DGX Spark systems for MXFP8 and NVFP4 data types" | — | NVIDIA tuned these dtypes for GB10 in cuBLAS 13.2 |
| **FP8 cuDNN attention** | Not optimized on cc 12.1 (cuDNN FP8 paths target cc 10.3 only) | — | Use SDPA or CUTLASS PR #3030 instead |
| **FP16 / BF16 cuDNN kernels** | cc 10.0 + 12.0 optimized in cuDNN; cc 12.1 inherits same path | — | Works, no special gotchas |
| **FlashAttention-4** | **DON'T** — requires tcgen05 + TMEM, physically absent from GB10 die | — | Silicon-level block. solatticus NVVM disassembly proof |

## Compile-flag rule of thumb

**Default for any cppmega GB10 build:**
```bash
nvcc -arch=sm_120f ...         # family variant, 9× faster than sm_121a for CUTLASS example 79
```

**Only use `sm_121a` if you specifically need an sm_121a-exclusive feature** — and as of CUTLASS 4.4.2 there are **none**.

**Never use `sm_120` / `sm_121` (no `a`/`f` suffix)** — strips architecture-specific features entirely, breaks block-scaled MMA (see PyTorch #172807 / #174161, llama.cpp #19662).

**Triton / torch.compile:**
```bash
export TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas
```
Required until Triton main picks up CUDA 13.x ptxas sm_12x support by default.

## CuTe DSL compat shim for sm_121a

Required only if you want FP4 paths. BF16/FP16 works without any patches. Add to the top of any cppmega CuTe DSL module targeting GB10:

```python
from cutlass.cute.nvgpu import warp as _warp
from cutlass.utils import smem_allocator as _smem

# (1) SmemAllocator capacity map: add sm_121 alias (one-line bug)
_smem.SMEM_CAPACITY_MAP.setdefault("sm_121", _smem.SMEM_CAPACITY_MAP["sm_120"])

# (2) FP4/NVF4/MXF4 warp MMA allowlist: include sm_121a (pure software omission)
if hasattr(_warp, "MmaSM120BlockScaledOp"):
    _warp.MmaSM120BlockScaledOp.admissible_archs = ["sm_120a", "sm_121a"]
    _orig_pi = _warp.MmaSM120BlockScaledOp.__post_init__
    def _patched(self):
        from cutlass.cutlass_dsl import CuTeDSL
        arch = CuTeDSL._get_dsl().get_arch_enum()
        if arch.major != 12:
            raise RuntimeError(f"SM120-family op requires sm_12x, got {arch}")
        # replay original dtype/shape checks inline
    _warp.MmaSM120BlockScaledOp.__post_init__ = _patched
```

Then invoke the allocator with `utils.get_smem_capacity_in_bytes("sm_120")` (hardcoded), as the shipped `blackwell_geforce/dense_gemm.py` already does at line ~227.

## Critical NVIDIA-staff quotes (archived)

- **@depaulmillz (CUTLASS, NVIDIA), cutlass#3144, 2026-04-05:** *"SM120 (RTX 6000, 5090, etc.) and SM121 (Spark) only support 99 KiB smem. B200 is SM100, which has 228 KiB shared memory. There is a fix for some carve-out calculations in SM120 kernels included in 4.4.2 and CUTLASS ToT which TRT-LLM has not updated to on ToT. We will work with the TRT-LLM team to update the version of CUTLASS they use."*
- **@depaulmillz, cutlass#3100:** *"SM121a and SM120a do not support `tcgen05` instead they utilize the warp level matrix multiply accumulate instructions."*
- **@mnicely (CUTLASS, NVIDIA), cutlass#2614, 2025-09-04:** *"We will absolutely support Spark. Let me confirm ETA and circle back."* (No follow-up as of 2026-04-11.)
- **@margaretz-nv (NVIDIA), dgx-spark-playbooks#22, 2025-12-19:** *"DSMEM, TMA/multicast are available and should work on Spark. tcgen05, TMEM are not supported on Spark."*
- **@pengbowang-nv (TRT-LLM, NVIDIA), TRT-LLM#11799, 2026-03-18:** *"AFAIK, trtllm-gen does not have support for SM120/121 yet. And SM100 kernels cannot run on SM120 directly. You may use XQA or FMHA_v2 as fallback."*
- **@johnnynunez, cutlass#2947, 2026-01-12:** *"tcgen05 is not in sm12x due space die and DLSS algorithm… Jetson Thor has tcgen05 with the same structure than gb200. In cutlass v4.4 we will add some specific MMA functions for sm12x"* (partially landed as `MmaSM120BlockScaledOp`; DSL tcgen05 still blocked.)
- **@masahi (Triton NVIDIA contributor), triton#8335:** *"sm120 / 121 does have MXFP support including mixed precision… Triton doesn't support mixed precision for sm120 MXFP yet, so things need to be emulated… sm120 is essentially the same as sm80 from the OGS kernel's perspective."*

## Dead paths — do not invest effort

1. **FlashAttention-4** — silicon-level block, no fix possible without new hardware
2. **CuTe Python DSL for tcgen05/UTCOMMA/UTCQMMA ops** — `admissible_archs = [sm_100a, sm_103a]`-level allowlists, NVVM binary verifier has 32-bit bitmask that doesn't cover sm_12x enum values. No NVIDIA timeline
3. **Building Triton from PR #9734 revision** — reverted because it broke `tensormap.replace`
4. **TRT-LLM `nvfp4_gemm_cutlass`** on pre-1.3.0rc2 + CUTLASS < 4.4.2 — silent numerical corruption
5. **B200 tile configs on GB10** — default 128×256×256 mainloop overflows 99 KiB smem; must re-tile
6. **FA2 / FA3 source-building for "speedup"** — matches SDPA perf, effort wasted
7. **tcgen05 / UMMA / any SM100 kernel binary on GB10** — won't load, no software shim
8. **FlashInfer trtllm-gen FMHA** — no SM12x cubins at all
9. **`CUTE_DSL_ARCH=sm_100a` / `sm_120a` runtime override on GB10** — cubin target check rejects at driver level

## Library status as of CUTLASS 4.4.2 / 2026-04

| Library | sm_121a status | Timeline |
|---|---|---|
| CUTLASS C++ API | SM121 kernels share major-code with SM120 since 4.2; SMEM carveout fix in 4.4.2; `sm_120f` is the right target | 4.4.2 released ~2026-03, bumped in TRT-LLM 2026-04-09 |
| CuTe Python DSL (nvidia-cutlass-dsl) | `BlockScaledMmaOp`, some `MmaF16BF16Op` variants, `_S2TCopyBase`, `Ld32x32bOp` reject sm_121a at admissible_archs level. **BUT** basic BF16/FP16 warp MMA + TMA + persistent scheduler work out of the box via `blackwell_geforce/dense_gemm.py` pattern | No ETA for DSL tcgen05. BF16 path works today |
| CUTLASS FlashAttention for SM120 (PR #3030) | BF16 CpAsync + TMA + FP8 inline-PTX, benchmarked on real GB10 | Unmerged, awaiting NVIDIA review |
| TensorRT-LLM | FP4 CUTLASS MoE broken < 4.4.2 on GB10 — workaround cuBLASLt path; TRITON MoE backend is the "use today" path; trtllm-gen FMHA has no SM120/121 cubins | CUTLASS 4.4.2 bump merged 2026-04-09; trtllm-gen no timeline |
| cuDNN | `cudnnGraphNotSupportedError` on NVFP4 mm_fp4 for SM120; BF16 paths work; SDPA via efficient-attention backend is fast | 9.13 / 9.17 tested — no GB10-specific kernels |
| Triton | `TRITON_PTXAS_PATH` fix; PR #9755 reverted the broken PR #9734. MoE/MXFP4 have gaps | Stable for BF16/FP16/FP32 |
| FlashInfer | FA2 prefill/decode/MLA all work; multiple PRs in flight for CuTe DSL / fmha_v2 / XQA backends | PR #2598, #3016, #2689 in review |
| Transformer Engine | No sm_121 PRs found; no documented sm_121-specific path. Works as of 2.14.0 on GB10 via generic sm_120 kernels. **Operational gotcha:** `transformer_engine_torch` ships as sdist-only and is always compiled against the torch currently in the venv — every torch-nightly bump risks an ABI break (`undefined symbol` at import). Recipe: `docs/transformer_engine_abi_rebuild.md` | Unknown |
| FlashAttention-2 / FA3 | Source-build workarounds documented (patched setup.py, CUTLASS 4.3+, flash_api.cpp arch check). Perf ≈ SDPA on GB10 | Community wheels exist |
| FlashAttention-4 | **Unrunnable** on GB10 — requires tcgen05/TMEM | Don't try |

## cppmega-specific implications

- **bwd_bwd optimization on GB10** — three parallel paths now viable:
  - (a) cuTile Python algorithm-first rewrite (minimal live set per scope, no memory tricks)
  - (b) **CuTe DSL BF16 hot-path** via `blackwell_geforce/dense_gemm.py` pattern (persistent scheduler + TMA bulk tensor + warp MMA + swizzled smem) — newly unblocked on sm_121a as of 2026-04-11
  - (c) Stay with TileLang baseline (167 µs, known reference)
  Run all three in parallel and compare on real kernel timing.

- **Training on GB10** — only for smoke tests. Bandwidth-bound at 273 GB/s LPDDR5X. **Production goes to H200 / B200.**

- **FP8 mamba kernels on GB10** — use cuBLAS 13.2 (officially tuned for Spark per release notes) or wait for CUTLASS PR #3030 merge. Don't hand-roll against `SM120_16x8x32_TN<e4m3,e4m3,f32>` atoms unless profiling demands it.

- **FA2 / FA3 on GB10** — skip entirely, SDPA is equivalent performance with zero build work.

## References

See `docs/gb10_sm121_hardware.md` for the full reference list.
