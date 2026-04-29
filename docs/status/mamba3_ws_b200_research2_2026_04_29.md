# Mamba3 WS / B200 Research 2

Date: 2026-04-29

Scope: research only. No production code changes. Sources were checked with Perplexity
Gemini 3.1 Pro / GPT-5.2 Thinking, Exa, Tavily, Brave, and direct inspection of
primary repositories/docs. Perplexity Deep Research timed out after 120s; Brave
returned exhausted-credit responses.

## Executive recommendations

1. For TileLang auto producer/consumer warp specialization, prototype only
   kernels whose hot loop is a `T.Pipelined(..., num_stages >= 1)` loop with at
   least one TMA-capable global-to-shared tile copy inside the loop. Use ordinary
   `T.copy` first and let TileLang classify it as TMA; use explicit `T.tma_copy`
   only when manual barriers are required.
2. Keep a Hopper fallback path. Modal explicitly says Blackwell has weaker
   precompiled-kernel coverage than Hopper, and `gpu="B200+"` can land on B300,
   which requires CUDA 13.0+.
3. For Modal long B200/B200+ jobs, treat every run as retryable: checkpoint to
   a `Volume`, write append-only per-run logs/metrics under unique paths, and
   store only small status/state entries in `Dict`. Avoid multi-writer writes to
   the same Volume file.
4. For B200/B300 build targets, include PTX and native cubins where available:
   B200/GB200 are compute capability 10.0 (`sm_100`), B300/GB300 are 10.3
   (`sm_103`). Do not assume `sm_100a`/family-specific binaries are portable.
5. For Mamba kernels, the most actionable upstream path is still
   `state-spaces/mamba` source, but there are useful design signals from the
   2026 Mamba-3 release: Triton for SISO prefill, TileLang for MIMO prefill, and
   CuTe DSL for decode on Hopper. For B200, expect to rebuild and profile rather
   than rely on wheels.

## 1. TileLang producer/consumer WS conditions

Primary source inspection used TileLang main at commit `4639c27` cloned to
`/tmp/tilelang-research`.

The pass is `tl.ProducerConsumerWarpSpecialized`, implemented in
`src/transform/producer_consumer_ws.cc`. It runs before layout inference and
`LowerTileOp`, classifies high-level tile ops, rewrites eligible loop bodies into
producer and consumer branches, inserts mbarriers, and converts TMA-eligible
copies into `tl.tileop.tma_copy` with barrier annotations.

Exact conditions observed in source:

- Global gate: target must be CUDA and have TMA support. `tilelang/engine/phase.py`
  returns false when target is not CUDA or `have_tma(target)` is false.
- Pass config gate: `tl.disable_warp_specialized` must be false. Public key is
  `tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED`.
- Manual WS gate: if the function already contains manual `T.ws()` /
  `warp_specialize` / `kWarpSpecializationScope`, the auto pass skips it.
- Candidate gate: the function body must contain a pipeline loop with a
  `num_stages` annotation whose integer value is `>= 1`.
- Candidate gate: inside that pipeline loop there must be at least one TMA
  producer tile op. For generic `T.copy`, `copy->CheckBulkLoad(target, ...)`
  must pass, `disable_tma` must not be set, and the direction must be
  global-to-shared. Explicit `T.tma_copy` is also treated as a producer only for
  valid global-to-shared TMA loads. TMA stores are kept on the consumer side.
- Layout gate: if the destination shared buffer has a layout annotation, the
  layout must be TMA-compatible: identity/linear or recognized 32B/64B/128B
  swizzle. Other layouts prevent the copy from becoming a TMA producer.
- Rewriter gate: the pass then looks for the block containing the pipeline loop,
  flattens the body, classifies statements, and requires `num_tma > 0`. If the
  pre-scan found a candidate but the rewriter does not actually transform it, it
  falls back to the original function and strips `num_stages` annotations to
  avoid broken non-WS TMA pipelining.

Negative examples confirmed by tests:

- `num_stages=0` does not auto warp-specialize and keeps ordinary `T.copy` on
  the synchronous path.
- cp.async-only loops do not auto warp-specialize even with `num_stages=1`.
- `num_stages=1` with a pure TMA-capable loop does auto warp-specialize.
- Mixed TMA plus cp.async with `num_stages=1` is tested as keeping auto-WS in
  current TileLang, even though the top-of-file v1 comments still mention pure
  TMA limitations.

Minimal auto-WS shape to use:

```python
@tilelang.jit(
    out_idx=[2],
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False},
)
def matmul_ws(M, N, K, block_M=128, block_N=128, block_K=64):
    @T.prim_func
    def main(A: T.Tensor((M, K), T.float16),
             B: T.Tensor((K, N), T.float16),
             C: T.Tensor((M, N), T.float16)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M),
                      threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])
    return main
```

Manual `T.tma_copy` shape when explicit barriers are needed:

```python
for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
    T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
    T.barrier_arrive(mbar_A)
    T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
    T.barrier_arrive(mbar_B)
    T.mbarrier_wait_parity(mbar_A, k % 2)
    T.mbarrier_wait_parity(mbar_B, k % 2)
    T.gemm(A_shared, B_shared, C_local)
```

Register annotations:

- `T.annotate_producer_reg_dealloc(reg_count=24)` is a wrapper around
  `T.set_max_nreg(reg_count, 0)`.
- `T.annotate_consumer_reg_alloc(reg_count=240)` is a wrapper around
  `T.set_max_nreg(reg_count, 1)`.
- `T.no_set_max_nreg()` can suppress automatic register allocation annotation.

Action for cppmega:

- Start with `num_stages=1,2,3` sweeps. `num_stages=1` is sufficient to fire
  auto-WS, but deeper stages increase shared-memory pressure.
- Keep annotated shared layouts linear or standard swizzles until TMA codegen is
  proven.
- Add a compile-time source check in probes: assert generated source contains
  `tma_load`/mbarrier and the WS branch shape; compare against
  `num_stages=0` as a negative control.

## 2. Modal B200/B200+ operational notes

Official Modal GPU docs list `B200` and `B200+`. `gpu="B200"` requests B200.
`gpu="B200+"` lets Modal run on B200 or B300, is billed as B200, and should only
be used when the code is compatible with both. Modal states B300 requires CUDA
13.0+.

GPU selection:

- B200/H200/H100/A100/L4/T4/L40S support up to 8 GPUs per container; requesting
  more than 2 GPUs usually increases wait time.
- Modal recommends understanding the bottleneck before jumping to B200. Small
  batch language model jobs are often memory-bound rather than arithmetic-bound.
- Hopper remains operationally safer when benchmarking libraries that may not
  yet ship Blackwell prebuilt kernels.

Long jobs:

- Default Modal Function timeout is 300s. User-configured timeout can be 1s to
  24h per execution attempt.
- Functions are preemptible by default. Modal restarts the same input after a
  preemption; long training/search jobs should checkpoint frequently and be
  idempotent.
- GPU Functions do not support `nonpreemptible=True`.

Volumes:

- Use a Volume for checkpoints, artifacts, generated kernel sources, profiling
  traces, and durable result logs.
- Volume writes become visible outside the current container after background
  commits or explicit `.commit()`. Other containers need `.reload()` to see
  committed changes.
- Volumes v1 are best below roughly 50k files and have a 500k inode limit.
  Volumes v2 support more files and many distinct-file writers, but remain beta.
- Avoid concurrent writes to the same file; Modal documents last-write-wins
  semantics and no distributed file locking.

Dicts:

- Use `modal.Dict` for small job state: run id, current stage, best config,
  status, heartbeat timestamp, artifact path.
- Do not use Dict for large logs or tensors. Modal recommends small objects
  under 5 MiB; per-object limit is 100 MiB and each entry expires after 7 days
  of inactivity.
- Mutable nested updates must be written back explicitly.

Logging pattern:

- Log to stdout/stderr for dashboard/CLI visibility.
- Also append structured JSONL to a Volume path such as
  `/runs/{run_id}/events.jsonl` and checkpoint `summary.json` periodically.
- Put only pointers and final status in Dict. This avoids Dict size/TTL limits
  and preserves results after retries/preemptions.
- CLI references: `modal app logs` streams app logs; `modal container logs`
  streams a specific running container. GPU health messages are injected into
  container log streams as `[gpu-health] ...` records.

## 3. Blackwell B200/B300 compatibility

NVIDIA's current CUDA GPU table lists:

- GB200 / B200: compute capability 10.0.
- GB300 / B300: compute capability 10.3.
- H100/H200/GH200: compute capability 9.0.
- RTX PRO / workstation Blackwell: compute capability 12.0; do not treat these
  as datacenter B200-compatible targets.

CUDA compatibility implications:

- Modal's `B200+` can allocate B300, so use CUDA 13.0+ if opting in.
- CUDA applications built with older toolkits can run on Blackwell only if they
  include compatible PTX for JIT; otherwise native cubins for Hopper (`sm_90`)
  are not enough for Blackwell.
- For native Blackwell builds, include `sm_100` for B200 and `sm_103` for B300
  when the toolchain supports it. Include PTX fallback for forward coverage.
- Avoid accidentally compiling only family-specific or architecture-specific
  forms that reduce portability. NVIDIA's Blackwell compatibility guide calls
  out that `sm_100a` / `compute_100a` architecture-conditional features are not
  forward/backward compatible in the same way as generic PTX/cubin targets.

Tuning implications:

- B200 compute capability 10.0 retains a Hopper-like CUDA programming model but
  adds Blackwell features. NVIDIA's tuning guide reports compute capability 10.0
  shared memory capacity up to 228 KB/SM and max shared memory per block around
  227 KB.
- Compute capability 12.0 workstation Blackwell has different occupancy/shared
  memory limits, so local RTX Blackwell results are not a clean proxy for B200.
- For B200/B300, explicitly report `nvidia-smi`, CUDA runtime, driver,
  `torch.version.cuda`, `nvcc --version`, and generated arch flags in every run.

## 4. Mamba2/Mamba3 kernel candidates beyond HF package defaults

Candidate shortlist:

| Candidate | Kernel stack | Hopper status | B200/B300 expectation | Recommendation |
| --- | --- | --- | --- | --- |
| `state-spaces/mamba` source | CUDA C++ selective scan plus Triton SSD/Mamba3 kernels | Primary upstream; Mamba-2 and Mamba-3 source paths exist | Rebuild from source; validate arch flags and B300 CUDA 13 | Baseline and oracle |
| Mamba-3 upstream kernels | Triton SISO prefill, TileLang MIMO prefill, CuTe DSL decode | Reported by authors on H100-SXM 80GB | No public B200 claim found; promising because TileLang/CuTe can target new archs | Highest-value research target |
| PyTorch fused Mamba2 SSD Triton work | Fuses five SSD kernels into one Triton kernel | Blog reports 1.50x-2.51x SSD speedups on A100/H100 | Blog says it does not yet use TMA/thread-block clusters/TMEM; good B200 opportunity | Mine for fusion strategy |
| Hugging Face `kernels-community/mamba-ssm` | packaged custom kernels | Useful install/reference path | Treat as package convenience, not enough for B300 compatibility | Secondary |
| JAX/XLA Mamba2 variants | shaped primitives, no custom CUDA/Triton | Portability-oriented | Less relevant to cppmega CUDA kernels | Low priority for B200 perf |
| Minimal Mamba3 PyTorch ports | pure PyTorch / educational | correctness/reference only | not performance path | Reference only |

Concrete Mamba takeaways:

- The official `state-spaces/mamba` README now documents Mamba-3 and says
  Mamba-3 should be installed from source with `MAMBA_FORCE_BUILD=TRUE`.
- The upstream repo exposes `mamba_ssm/modules/mamba3.py` and
  `mamba_ssm/ops/triton/mamba3/*`.
- The Mamba-3 release says kernels are open-sourced and built with Triton,
  TileLang, and CuTe DSL; it reports H100-SXM 80GB latency tables. It does not
  establish B200/B300 performance.
- The PyTorch fused Mamba2 SSD blog identifies a useful next kernel target:
  their fused Triton kernel is faster on A100/H100 but explicitly does not use
  Hopper TMA/thread-block clusters or Blackwell TMEM. That gap maps directly to
  cppmega/TileLang B200 research.

Action for cppmega:

- First reproduce upstream Mamba3 SISO/MIMO on H100/H200 or B200 with source
  builds. Record whether failures are build-arch, Triton codegen, TileLang, or
  runtime correctness.
- Prioritize Mamba3 MIMO prefill TileLang kernels as the bridge to producer /
  consumer WS and TMA experiments.
- Treat decode separately: Mamba-3 authors used CuTe DSL for decode because
  decode wants lower-level layout/warp-specialization control.
- Do not spend time on pure-PyTorch Mamba3 implementations except as numerical
  references.

## Source links

TileLang:

- https://github.com/tile-ai/tilelang/blob/main/src/transform/producer_consumer_ws.cc
- https://github.com/tile-ai/tilelang/blob/main/tilelang/engine/phase.py
- https://github.com/tile-ai/tilelang/blob/main/tilelang/language/loop.py
- https://github.com/tile-ai/tilelang/blob/main/tilelang/language/copy_op.py
- https://github.com/tile-ai/tilelang/blob/main/tilelang/language/builtin.py
- https://github.com/tile-ai/tilelang/blob/main/testing/python/transform/test_tilelang_transform_producer_consumer_ws.py
- https://github.com/tile-ai/tilelang/blob/main/testing/python/issue/test_tilelang_issue_tma_no_ws.py
- https://github.com/tile-ai/tilelang/blob/main/testing/python/language/test_tilelang_language_tma_copy.py
- https://github.com/tile-ai/tilelang/blob/main/examples/deepseek_mla/README.md
- https://github.com/tile-ai/tilelang/commit/ded6a992218a5ec0dbb490128cb21aae7794cdb8

Modal:

- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/timeouts
- https://modal.com/docs/guide/preemption
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/dicts
- https://modal.com/docs/reference/cli/app
- https://modal.com/docs/reference/cli/container
- https://modal.com/docs/guide/gpu-health/

NVIDIA / CUDA:

- https://developer.nvidia.com/cuda/gpus
- https://docs.nvidia.com/cuda/archive/13.0.2/blackwell-compatibility-guide/index.html
- https://docs.nvidia.com/cuda/archive/13.2.0/blackwell-compatibility-guide/index.html
- https://docs.nvidia.com/cuda/archive/13.2.0/blackwell-tuning-guide/index.html
- https://developer.nvidia.com/blog/nvidia-blackwell-and-nvidia-cuda-12-9-introduce-family-specific-architecture-features

Mamba:

- https://github.com/state-spaces/mamba
- https://github.com/state-spaces/mamba/tree/main/mamba_ssm/ops/triton/mamba3
- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/mamba3.py
- https://github.com/state-spaces/mamba/blob/main/csrc/selective_scan/selective_scan_fwd_kernel.cuh
- https://github.com/state-spaces/mamba/blob/main/mamba_ssm/ops/triton/ssd_chunk_scan.py
- https://huggingface.co/kernels-community/mamba-ssm
- https://www.together.ai/blog/mamba-3
- https://arxiv.org/abs/2603.15569
- https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/
- https://arxiv.org/abs/2603.09555
