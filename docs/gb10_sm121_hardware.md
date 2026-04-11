# GB10 (sm_121a) / consumer Blackwell hardware capabilities

Reference for what NVIDIA Blackwell **consumer** silicon (sm_120a RTX 5090 and sm_121a GB10 DGX Spark) actually supports at the PTX/SASS level, for deciding which cppmega kernels to target and how to compile them. Compiled from NVIDIA CUTLASS source, PTX ISA 8.7/8.8, NVIDIA CUTLASS issues #2800/#2947/#3044/#3100/#3144, TRT-LLM #11368/#11799, NVIDIA DevTalk DGX Spark threads, and empirical verification on real GB10 hardware (2026-04-11).

## TL;DR — what exists on sm_121a

- **Full warp-level `mma.sync`** family (Ampere/Hopper-era, plus Blackwell extended kinds)
- **TMA** (`cp.async.bulk.tensor`, single-CTA) — from Hopper, preserved
- **Swizzled shared memory** (`ldmatrix` / `stmatrix` / `stsm`)
- **FP4 / FP6 / FP8 / BF16 / FP16 / TF32** datatypes via extended `mma.sync`
- **Block-scaled MMA** (`kind::mxf8f6f4.block_scale`, `kind::mxf4nvf4.block_scale`) — BUT only when the compile target has the `a` suffix (`sm_120a`, `sm_121a`, NOT bare `sm_120` / `sm_121`)
- **Warp specialization** via `setmaxnreg.inc/.dec` (PTX 8.8 family-features)
- **Clusters** via `clusterlaunchcontrol.try_cancel` — but WITHOUT 2-SM multicast

## TL;DR — what does NOT exist on sm_121a

- **`tcgen05.*` family** (tcgen05.mma, tcgen05.ld, tcgen05.st, tcgen05.alloc, tcgen05.cp) — **datacenter-only** (sm_100a/sm_101a/sm_103a/sm_110a). Silicon absent from the GB10 die
- **Tensor Memory (TMEM, 256 KiB/SM)** — silicon not present
- **`wgmma.mma_async` (Hopper WGMMA)** — deprecated on all Blackwell (both datacenter and consumer)
- **2-SM UMMA / TMA multicast** — sm_100a only
- **UMMA SASS family** — no; sm_120/121 emits `HMMA` / `QMMA` families same as sm_90 non-WGMMA paths

## Capability matrix

| Feature | sm_100a (B200) | sm_121a (GB10) | sm_120a (RTX 5090) | sm_90a (H100) |
|---|---|---|---|---|
| `tcgen05.mma` | ✅ | ❌ | ❌ | ❌ |
| `tcgen05.ld / .st / .alloc / .cp` | ✅ | ❌ | ❌ | ❌ |
| TMEM (Tensor Memory, 256 KiB/SM) | ✅ | ❌ silicon absent | ❌ silicon absent | ❌ |
| `wgmma.mma_async` | ❌ deprecated | ❌ | ❌ | ✅ |
| `cp.async.bulk.tensor` (TMA, single-CTA) | ✅ | ✅ | ✅ | ✅ |
| TMA multicast (cluster > 1) | ✅ | ⚠️ (1) | ⚠️ (1) | ✅ |
| DSMEM (distributed shared memory) | ✅ | ✅ | ✅ | ✅ |
| Extended `mma.sync kind::f8f6f4` | ✅ | ✅ | ✅ | — |
| Extended `mma.sync kind::mxf8f6f4.block_scale` | ✅ | ✅ (**`a` suffix required**) | ✅ (**`a` suffix required**) | — |
| Extended `mma.sync kind::mxf4nvf4.block_scale` | ✅ | ✅ (`a` suffix required) | ✅ (`a` suffix required) | — |
| `ldmatrix` / `stmatrix` swizzled smem | ✅ | ✅ | ✅ | ✅ |
| `setmaxnreg.inc/.dec` warp specialization | ✅ | ✅ | ✅ | ✅ |
| Cluster launch control | ✅ | ✅ | ✅ | ✅ |
| Shared memory per SM (physical) | 228 KiB | **~128 KiB** | 128 KiB | 228 KiB |
| Shared memory per SM (CUTLASS dynamic budget) | 232 KiB | **99 KiB** (`sm120_smem_capacity_bytes = 101376`) | 99 KiB | 228 KiB |
| Max resident warps / SM | 64 | **48** | 48 | 64 |
| 32-bit registers / SM | 64 K | 64 K | 64 K | 64 K |
| Max registers / thread | 255 | 255 | 255 | 255 |

(1) @margaretz-nv (NVIDIA) says DSMEM + TMA multicast available on Spark; CUTLASS example 79 forces `cluster_shape=1×1×1` with comment "GeForce does not support multicast feature of TMA load". Probable reconciliation: multicast instruction exists but cluster-size limit makes it a no-op. **Do not rely on multicast speedup on sm_120/121.**

## Toolchain requirements

- **PTX ISA 8.8** minimum for `.target sm_121a` / `.target sm_121`. PTX 8.7 errors "version 8.7 does not support .target sm_121a". Current CUDA 13.2 pairs with PTX ISA 9.2.
- **CUDA 12.9** earliest shipping nvcc with sm_121; CUDA 13.2 is current.
- **Driver ≥ 595.45.04** for CUDA 13.2.
- **PTX ISA 8.7** introduced sm_120 / sm_120a; **PTX ISA 8.8** introduced sm_121, sm_121a, and family targets `sm_100f`, `sm_101f`, `sm_103f`, `sm_120f`, `sm_121f`.

### Architecture suffix: `a` vs `f` vs no suffix (verbatim from NVCC driver docs §1.1)

- **`sm_XY`** (no suffix) — baseline, forward-compatible. Compiles to all `sm_MN` / `sm_MNa` / `sm_MNf` where MN ≥ XY. **Does NOT enable architecture-specific features.**
- **`sm_XYf`** (family-specific) — enables features common across the GPU family. Compiles to `sm_XZ` / `sm_XZf` / `sm_XZa` where Z ≥ Y and same family. This is the right choice when you want the optimizations to work across sm_120a / sm_121a / sm_120f / sm_121f without binary-locking.
- **`sm_XYa`** (architecture-specific) — enables the complete set of architecture-specific features. **Binary-locked** to the exact compute capability. PTX with `.target sm_XYa` can only be compiled to `sm_XYa`.

### CRITICAL — always use `a` or `f` suffix on sm_12x

Several downstream bugs come from stripping the `a` suffix:
- **PyTorch #172807 / #174161** — AOTInductor emits PTX with `.target sm_120a` but nvcc is called with `-arch=sm_120` (stripped). Block-scaled MMA then fails.
- **llama.cpp #19662** — built with `-arch=sm_120`, errors at `"Instruction 'mma with block scale' not supported on .target 'sm_120'"`.
- **CUTLASS #3044** — Python CuTe DSL `MmaAtomSM80Type.get()` segfault on sm_121a for FP8 paths; the PTX path works, the DSL lowering doesn't. Workaround: use C++ CUTLASS directly for FP8 on GB10.

**Rule for cppmega GB10 builds:** always compile with `-arch=sm_121a` OR `-arch=sm_120f` (family variant, covers both sm_120a and sm_121a). **Never** use bare `sm_120` / `sm_121`. Verify via `cuobjdump --dump-ptx` that emitted PTX `.target` has the `a` or `f` suffix before shipping any C++ / Triton / torch.compile kernel.

### `-arch=sm_120f` vs `-arch=sm_121a` matters 9× in practice

Empirical: CUTLASS example 79 `blackwell_geforce_gemm` on GB10 hardware:
- `-arch=sm_121a`: **37-41 TFLOPS** (per TRT-LLM issue #11368)
- `-arch=sm_120f`: **356 TFLOPS** (71% of peak sparse, per NVIDIA forum "SM121 CUTLASS Kernel Optimization Results")

The `sm_121a` restrictive path strips family-common optimizations; the `sm_120f` family variant keeps them. **Default to `sm_120f` for GB10 unless there's a specific sm_121a-exclusive feature needed** — and as of CUTLASS 4.4.2 there are none.

## CUTLASS atom inventory for sm_120/121 (4.4.2)

Total **~160 atom specializations** across two files covering dense + sparse:

**`include/cute/arch/mma_sm120.hpp` (dense, 80 atoms, 3278 lines):**

| PTX `.kind::` modifier | Shape | Dtype inventory |
|---|---|---|
| `f8f6f4` | m16n8k32 | 25 pairs of {e5m2, e4m3, e3m2, e2m3, e2m1} × {f16, f32} accum = 50 |
| `mxf8f6f4.block_scale.scale_vec::1X` | m16n8k32 | Same 25 dtype pairs + `ue8m0` scale = 25 |
| `mxf4nvf4.block_scale.scale_vec::2X` | m16n8k64 | e2m1 × e2m1 + `ue8m0` scale |
| `mxf4nvf4.block_scale.scale_vec::4X` | m16n8k64 | e2m1 × e2m1 + `ue8m0` or `ue4m3` scale |

**`include/cute/arch/mma_sm120_sparse.hpp` (2:4 sparse, 80 atoms, 3467 lines):**

| PTX `.kind::` modifier | Shape | Notes |
|---|---|---|
| `f8f6f4.sp::ordered_metadata` | m16n8k64 | 25 dtype pairs × {f16, f32} accum |
| `mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X` | m16n8k64 | `ue8m0` |
| `mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::{2X,4X}` | m16n8k128 | `ue8m0` or `ue4m3` |

**Config guards:** `CUTE_ARCH_F8F6F4_MMA_ENABLED`, `CUTE_ARCH_MXF8F6F4_MMA_ENABLED`, `CUTE_ARCH_MXF4NVF4_{2X,4X}_{UE8M0,UE4M3}_MMA_ENABLED`. All active when `CUTLASS_ARCH_MMA_SM120A_ENABLED || SM121A_ENABLED`.

**No `mma_sm121.hpp` exists** — sm_121a reuses the SM120 atoms via the config alias. The `cutlass::arch::Sm120` ArchTag covers both targets; `cutlass::arch::Sm121` does not exist.

**Standard mma.sync (BF16/FP16/TF32/INT8/INT4)** — inherited from `mma_sm80.hpp` (Ampere family). Compiler auto-selects on sm_121a targets. **No sm_121-specific BF16 atom** — our cppmega BF16 mamba3 kernels already use the optimal MMA atom path on GB10.

## SASS opcodes emitted on sm_120/sm_121

| PTX instruction | SASS opcode | Notes |
|---|---|---|
| `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | `HMMA.16816.F32.BF16` | **Same as sm_90 non-WGMMA** — Ampere-era HMMA family |
| `mma.sync.aligned.m16n8k16.row.col.f16.f16.f16` | `HMMA.16816.F16.F16` | f16 accumulator — 2× throughput of f32 accum |
| `mma.sync.aligned.m16n8k32.kind::f8f6f4` | `QMMA.*` / `HMMA.MXF*` | Blackwell-specific extended |
| `mma.sync.aligned.m16n8k32.kind::mxf8f6f4.block_scale` | `QMMA.MXF*` with scale fetch | |
| `mma.sync.aligned.m16n8k64.kind::mxf4nvf4.block_scale` | `QMMA.MXF4.*` | |

**No `BlackwellMMA` / `UMMA` SASS family on sm_120/sm_121.** UMMA is reserved for sm_100a where TMEM exists.

**Tensor core instruction terminology (from NVIDIA DevTalk):**
- **OMMA** = warp-level FP4 MMA on sm_120/121 (present)
- **QMMA** = warp-level FP8 MMA on sm_120/121 (present)
- **UTCOMMA** = TMEM-coupled FP4 MMA on sm_100 (absent on sm_12x)
- **UTCQMMA** = TMEM-coupled FP8 MMA on sm_100 (absent on sm_12x)

**Common misconception to avoid:** "FP4 is blocked on GB10." → False. FP4 *is* available on sm_121 via warp-level OMMA. What's blocked is the *tcgen05* code path (UTCOMMA). Most CUTLASS FP4 kernels hard-code the tcgen05 path which is why "FP4 doesn't work on consumer Blackwell" became folk wisdom. CUTLASS example 79 is the OMMA-based reference that *does* work on GB10.

## Peak performance ceiling

From NVIDIA RTX Blackwell whitepaper + dev forum confirmations (Crovella) + TRT-LLM #11368 + NVIDIA forums "SM121 CUTLASS Kernel Optimization Results":

| Chip | BF16 (f32 acc) | FP16 (f16 acc) | FP8 | FP4 | SMEM | Memory BW |
|---|---|---|---|---|---|---|
| GB10 DGX Spark (sm_121a) | ~100 TFLOPS | ~200 TFLOPS | ~200 TFLOPS | ~400 TFLOPS (1 PFLOP sparse spec) | ~128 KB | **273 GB/s** LPDDR5X |
| RTX 5090 (sm_120a GB202, 170 SM) | 209 TFLOPS | **419 TFLOPS** | 838 TFLOPS | 1676 TFLOPS | 128 KB | 1792 GB/s GDDR7 |
| B200 (sm_100a) | 2250 TFLOPS | — | 4500 TFLOPS | 9000 TFLOPS | 228 KB | 8000 GB/s HBM3e |
| H100 (sm_90a) | 989 TFLOPS | — | 1979 TFLOPS | — (no FP4) | 228 KB | 3350 GB/s HBM3 |

**FP16 with f16 accumulator is 2× faster than BF16 with f32 accumulator on sm_120a** (419 vs 209 TFLOPS). If numerical tolerance permits, this gives free 2× speedup for GEMM-heavy kernels.

**Implications for cppmega:**
- H100 peak BF16 = 989 TFLOPS vs GB10 peak BF16 = ~100 TFLOPS → **10× absolute perf gap**
- Our TileLang bwd_bwd 167 µs baseline on GB10 is already close to what the HW can deliver
- The cuTile 633 µs is 3.8× below that — this is a kernel scheduling problem, not an instruction problem
- GB10 is **bandwidth-bound** for LLM inference: 70B-FP4 gives 5.2 tok/s, 70B-FP8 gives 2.7 tok/s (via insiderllm.com teardown). Not a training target — use only for smoke tests and kernel validation

## The "Blackwell is two different ISAs" surprise

NVIDIA compute-capability versioning skipped 11 entirely — consumer Blackwell jumped from sm_89 (Ada) to sm_120/121 directly. Per NVIDIA forum rep (via jangwook.net): DGX Spark / GB10 tensor cores are **"closer to the GeForce Ampere-style MMA model"** — RT cores + DLSS silicon took die budget that would have gone to TMEM + tcgen05 on datacenter chips.

Despite the "Blackwell" marketing umbrella, sm_120/121 is architecturally closer to **Ampere+FP4 bolted on** than to B200. FlashAttention-4, FlashMLA SM100 backend, and most DeepSeek / FlashInfer SM100 kernel paths simply cannot target sm_120/121 and never will.

## References

- NVIDIA CUTLASS issues: [#2800](https://github.com/NVIDIA/cutlass/issues/2800), [#2947](https://github.com/NVIDIA/cutlass/issues/2947), [#3044](https://github.com/NVIDIA/cutlass/issues/3044), [#3100](https://github.com/NVIDIA/cutlass/issues/3100), [#3144](https://github.com/NVIDIA/cutlass/issues/3144)
- NVIDIA CUTLASS PR [#3030](https://github.com/NVIDIA/cutlass/pull/3030) — SM120 FlashAttention BF16 CpAsync + TMA + FP8 inline-PTX (unmerged, usable)
- NVIDIA TRT-LLM issues: [#11368](https://github.com/NVIDIA/TensorRT-LLM/issues/11368), [#11799](https://github.com/NVIDIA/TensorRT-LLM/issues/11799)
- PyTorch issues: [#172807](https://github.com/pytorch/pytorch/issues/172807), [#174161](https://github.com/pytorch/pytorch/issues/174161)
- llama.cpp issue: [#19662](https://github.com/ggml-org/llama.cpp/issues/19662)
- NVIDIA DevTalk: ["SM121 CUTLASS Kernel Optimization Results" (NVFP4 356 TFLOPS on DGX Spark)](https://forums.developer.nvidia.com/t/sm121-cutlass-kernel-optimization-results-nvfp4-356-tflops-moe-grouped-gemm-on-dgx-spark/359960), ["Dearest CUTLASS TEAM... tcgen05 FP4 support for DGX Spark"](https://forums.developer.nvidia.com/t/dearest-cutlass-team-when-the-hell-are-you-going-to-properly-fix-tcgen05-fp4-support-for-dgx-spark-gb10-sm121/359598)
- NVIDIA CUDA: [Programming Guide Compute Capabilities Table](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/compute-capabilities.html), [Blackwell Tuning Guide](https://docs.nvidia.com/cuda/blackwell-tuning-guide/index.html), [NVCC Driver 13.2 docs](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/), [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Colfax Research CUTLASS Tensor Memory tutorial](https://research.colfax-intl.com/cutlass-tutorial-writing-gemm-kernels-using-tensor-memory-for-nvidia-blackwell-gpus/)
- [gau-nernst tcgen05 blog](https://gau-nernst.github.io/tcgen05/)
- [solatticus FA4 sm_120 investigation gist](https://gist.github.com/solatticus/aab6ec3a0436748b021cbbdd12e8c739)
- [SemiAnalysis: Dissecting NVIDIA Blackwell](https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor)
- [arXiv 2512.02189 — Microbenchmarking NVIDIA Blackwell](https://arxiv.org/abs/2512.02189)
