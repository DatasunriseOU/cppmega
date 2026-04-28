# Cppmega Run Profiles And Token Flow

Status: canonical
Last updated: 2026-04-28
Scope: typed run-profile contract, local GB10 token flow, and precision/layout
boundaries for the current NAM56R-quarter debug lane.

## Run-Profile Contract

Core launch choices live in `cppmega/recipes/run_profiles.py`.

Each profile is built by a setter function that fills a `RunProfile` dataclass:

- `set_local_gb10_quarter_profile()`:
  single-GB10 correctness/profiling lane, 13 layers at NAM56R width,
  real 4k clang data, MTP depth 2, CCE MTP CE, tensorwise FP8, patched FA4
  SM120 attention routing, no-master Muon, q8 Muon momentum, local DDP
  contiguous grad buffer disabled, and explicit source roots for the local
  `/home/dave/flash-attention-fa4` plus `/home/dave/TransformerEngine` forks.
- `set_h200_dsa_9_4_m_profile()`:
  full-depth H200 production-target skeleton. Remote scripts still own
  machine-specific PP/VPP/EP orchestration, but the model-level contract is
  typed here so local GB10 exceptions do not leak into H200 defaults.

Launchers should render a named profile instead of hand-setting core knobs:

```bash
RUN_PROFILE="${RUN_PROFILE:-local_gb10_quarter}"
eval "$(PYTHONPATH="${ROOT}:${MEGATRON_ROOT}:${PYTHONPATH:-}" \
  python -m cppmega.recipes.run_profiles shell "${RUN_PROFILE}")"
```

The shell exports remain only because current shim modules read `os.environ` at
import time. The source of truth is the dataclass, not the shell defaults. MTP
depth, CE route, FP8 recipe, optimizer, sequence length, batch sizes, and
derived Megatron args are profile fields.

Ad-hoc smoke changes are profile parameters, not new env defaults:

```bash
bash scripts/local_gb10_quarter_train.sh --train-iters 1 --mem-profile
bash scripts/local_gb10_quarter_train.sh --fp8-recipe off --train-iters 4
bash scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --param-storage mxfp8 \
  --mxfp8-bwd-backend flashinfer_cutlass \
  --mxfp8-transpose-emit-backend te \
  --mxfp8-transpose-emit-swizzled \
  --mxfp8-transpose-emit-strict
bash scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend te_tn_adapter \
  --mxfp8-transpose-emit-backend te \
  --mxfp8-transpose-emit-swizzled \
  --mxfp8-transpose-emit-strict
bash scripts/local_gb10_quarter_train.sh \
  --fp8-recipe mxfp8 \
  --mxfp8-bwd-backend cutlass_native \
  --mxfp8-transpose-emit-backend off \
  --cutlass-mxfp8-scale-backend compact
```

Those flags are forwarded to `cppmega.recipes.run_profiles shell ...` and mutate
the selected `RunProfile` after its setter runs.
The rendered `CPPMEGA_*` values are an internal transport for import-time
shims only; operator-facing precision/backend choices must be added as
`RunProfile` dataclass fields plus CLI overrides instead of documented as raw
environment-variable recipes.

`optimizer.param_storage` is the optimizer/model-storage contract:

- `auto`: BF16 params for BF16/tensorwise runs, primary MXFP8 params for
  `fp8_recipe=mxfp8`.
- `bf16`: keep ordinary BF16 model params.
- `mxfp8`: initialize TE module weights as primary MXFP8 tensors through
  Megatron `fp8_param`, without distributed `--fp8-param-gather`.

The local no-master Muon path owns this contract.  `--mxfp8-param-storage`
is intentionally separate from `--fp8-param-gather`: the former is local
authoritative MXFP8 storage; the latter is Megatron's distributed/FSDP
parameter all-gather contract.

Remote H200 launchers use the same helper contract for derived MTP/MoE/DSA
arguments.  The unsafe historical pattern was:

```text
render --mtp-num-layers 1, then sed it to MTP_DEPTHS
```

That is no longer allowed.  `MTP_DEPTHS` must be passed into the helper as
`mtp_num_predictors`, and EP/dispatcher/router-dtype choices must be rendered by
`build_nam56r_megatron_native_args()` / `build_megatron_args_bundle()` rather
than by string replacement.  `tests/test_remote_profile_contract.py` enforces
that remote launchers do not rewrite `--mtp-num-layers` after rendering.

MoE dispatcher contract:

- `NATIVE_ARGS` owns the single `--moe-token-dispatcher-type` flag. Launchers
  must not append a second dispatcher flag after rendering the typed profile.
- `alltoall` is the safe default and the local GB10 `TP=EP=1` lane.
- `flex` is explicit only: set `moe_expert_model_parallel_size > 1`,
  `moe_router_dtype=fp32`, and render `--moe-flex-dispatcher-backend`
  (`deepep` or `hybridep`). Megatron raises if the selected backend is missing;
  there is no silent fallback to alltoall.
- `cppmega.megatron.moe_dispatcher_patch` skips identity local-expert chunk
  sorts in the alltoall `TP=EP=1` case, removing no-op fused sort launches from
  local GB10 MoE layers. Set `CPPMEGA_MOE_SKIP_IDENTITY_CHUNK_SORT=0` for A/B
  profiling.

## Local GB10 Quarter Shape

Current `local_gb10_quarter` profile:

```text
B = global batch = micro batch = 4
S = sequence length = 4096
H = hidden size = 3584
Heads = 28
V = vocabulary size = 65536
Main layers = 13
Pattern = *EME*EME*EMM*
MTP suffix = /*-/*-
MTP predictors = 2
```

The same profile field, `model.mtp_depths=2`, drives both:

```text
hybrid_layer_pattern = *EME*EME*EMM*/*-/*-
Megatron arg        = --mtp-num-layers 2
```

This matters because older shell snippets could create two MTP suffixes while
still emitting `--mtp-num-layers 1`, which invalidated the test.

## Token Flow Diagram

```text
tokens
  int64 [B=4, S=4096]
  |
  | embedding lookup
  v
hidden
  BF16 [B, S, H=3584]
  |
  +--> optional ngram hash / structure features
  |      ngram table contributes compact BF16 feature vectors
  |      structure path is enabled as "core"
  |
  v
13-layer hybrid trunk: * E M E * E M E * E M M *

* = attention/MLA/DSA layer
  input:  BF16 [B, S, H]
  qkv/mla projections: TE Linear under BF16/tensorwise-FP8 recipe
  full attention:
    local GB10 defaults to attention_backend=flash and puts the patched
    `/home/dave/flash-attention-fa4` tree first on PYTHONPATH.  This avoids
    the mixed venv/source import path where `flash_attn.__init__` came from the
    installed 2.8.3 wheel while SM120 CUTE modules came from the source tree.
    Use `--attention-backend auto` only for explicit fallback repros.
  DSA path:
    indexer q/k accumulation:
      current accepted path keeps BF16 operands and accumulates into
      FP32/BF16-selected score storage [B, S_q, S_k] before top-k
    sparse attention:
      TileLang SparseMLA replaces unfused dense attention and avoids the huge
      [B * num_heads, S_q, S_k] full attention matrix
    target:
      streaming top-k emits only [B, S_q, topk] indices/scores
  kernel accumulators:
    Flash/TileLang reductions use FP32 registers/accumulators internally
    but do not expose persistent FP32 model tensors
  output: BF16 [B, S, H]

E = MoE block
  input:  BF16 [B, S, H]
  router logits/probs:
    FP32 when DeepEP/flex is active because DeepEP weighted dispatch expects
    FP32 top-k weights/probabilities
  token dispatch:
    local GB10 EP=1 uses alltoall; H200 EP lanes use alltoall unless a typed
    profile explicitly selects flex with deepep/hybridep and fp32 router probs
  expert GEMMs:
    BF16/tensorwise-FP8 TE GEMM path depending profile and hardware
  output: BF16 [B, S, H]

M = Mamba3 / M2RNN-family block
  input:  BF16 [B, S, H]
  in/out projections:
    TE Linear under BF16/tensorwise-FP8 recipe
  scan/state math:
    persistent state tensors are BF16 where accepted
    scalar A/dt/D style math may promote to FP32 inside kernels or local ops
  output: BF16 [B, S, H]

MTP predictor 1
  hidden in: BF16 [B, S, H]
  MTP local layer stack: attention + MLP over shifted targets
  CE route: CCE fused linear CE for local GB10 profile
  logged loss: mtp_1 loss

MTP predictor 2
  hidden in: BF16 [B, S, H]
  same local contract as predictor 1
  CE route: CCE fused linear CE
  logged loss: mtp_2 loss

main output head / CE
  hidden: BF16 [B, S, H]
  output weight: BF16 [V=65536, H=3584]
  local GB10 main head route:
    native LCE rejects cc=12.1, so Apple CCE fallback is used
  logits:
    fused CE avoids materializing full [B*S, V] logits as a persistent tensor
  reductions:
    loss reductions use FP32 internally where the fused CE kernel requires it
```

## Precision Boundaries

Persistent storage in the local tensorwise/BF16 profile:

- model parameters: BF16 after Megatron `Float16Module`
- ordinary hidden activations: BF16 between blocks
- optimizer main params: no FP32 master in the local no-master Muon lane
- Muon momentum: q8 data plus block absmax metadata
- fallback optimizer state: low-memory state selected by the profile

Persistent storage in the local MXFP8 profile:

- TE dense Linear / LayerNormLinear / GroupedLinear weights are primary MXFP8
  tensors through `--mxfp8-param-storage`, not BF16 tensors with an external
  sidecar and not distributed `--fp8-param-gather` storage.
- TE MXFP8 weights keep both rowwise and compact columnwise payload/scale
  storage for GEMM use. The latest storage receipt still includes BF16 model
  storage for non-covered tensors.
- Ordinary hidden activations between blocks, CE inputs, non-TE parameters,
  Mamba scan/state tensors, DSA/indexer tensors, and the BF16 outputs of dense
  GEMMs remain BF16.
- Covered non-FSDP Linear backward edges save GEMM-ready MXFP8
  rowwise-transposed operands in autograd instead of saving the BF16 Linear
  input. That saved operand is real materialized MXFP8 storage, not a persistent
  producer-side sidecar.

Non-persistent higher precision:

- attention and CE reductions can use FP32 accumulators/registers
- DSA/router score paths may use FP32 scores where API or numerical stability
  requires it
- Mamba scan scalar math can promote local A/dt/D values to FP32

MXFP8 is not the default local training path. It is a probe path controlled by
`PrecisionProfile.fp8_recipe="mxfp8"` plus the MXFP8 backward fields. The profile
must set those coherently; a TE precision config without the rendered
`fp8_recipe=mxfp8` profile value is rejected by the local launcher.

The accepted local MXFP8 Linear backward probe path no longer saves a BF16
Linear input plus a producer-side MXFP8 transpose sidecar for the covered
non-FSDP case. Local TE `Linear` and `LayerNormLinear` now prefer a GEMM-ready
MXFP8 rowwise-transposed saved operand inside autograd. Parameter quantization
also defers transpose creation instead of attaching forward-side sidecars. This
currently applies only when no backward input all-gather, CPU offload, or FSDP
scatter is involved. The full-model GB10 path still has an on-demand
copy-transpose bridge through `_cppmega_mxfp8_colwise_as_rowwise_transpose`
when only compact columnwise MXFP8 storage is available.

## GB10 MXFP8 Layout Boundary

GB10 can execute native block-scaled GEMM, and TE 2.16.dev exposes scale
swizzling APIs. The current mismatch is not lack of swizzling support. It is the
contract between three representations:

```text
TE compact rowwise tensor:
  payload: [rows, cols]
  scales:  [rows, ceil(cols / 32)]

TE compact columnwise tensor:
  payload: [rows, cols]
  scales:  [ceil(rows / 32), cols]

CUTLASS SM120 native block-scaled GEMM:
  expects operand/layout-compatible payload tiles
  expects SM1xx swizzled scale layout for stock TMA scale descriptors
```

The current TN adapter needs compact rowwise/columnwise payloads because it
retargets unsupported GB10 `NN`/`NT` backward GEMMs into supported `TN` GEMMs.
When the original operand is only available as TE compact columnwise storage,
stock CUTLASS/TMA cannot directly treat that byte layout as the transposed
native operand. The no-sidecar direct path therefore falls back to a manual
producer-warp payload/scale loader, which is correct but slow.

Short-term performance path:

- keep attention on the patched FA4 SM120 path; the following bullets are about
  dense MXFP8 Linear GEMMs only, not attention
- keep TE as the MXFP8 payload owner
- use TE/autograd saved rowwise-transpose operands for covered Linear backward
  edges
- convert only compact rowwise scales to FlashInfer/CUTLASS `layout_128x4`
  via the cppmega producer kernel
- call FlashInfer/CUTLASS SM120 GEMM for clean MXFP8 GEMMs
- keep direct no-sidecar only as a correctness/coverage path

Long-term path:

- write/own a narrow SM120 mainloop that consumes TE compact layout directly
- replace scalar producer-warp byte loops with vectorized/coalesced or cp.async
  style copies into the native shared-memory layout
- keep acceptance tests requiring zero BF16 fallback before calling the path
  accepted

## Validation Receipt

Previous local MTP=2 + Liger CE smoke:

```text
initial MTP=2 Liger log: /home/dave/logs/gb10_mtp2_liger_smoke_20260426_2115.log
dataclass launcher log: /home/dave/logs/gb10_profile_dataclass_smoke_20260426_212558.log
steps: 10 through the dataclass-rendered launcher
MTP: mtp_num_layers = 2, mtp_1 and mtp_2 losses logged
NaN iterations: 0
peak torch max allocated: 28392.57 MB
peak torch max reserved: 30526.00 MB
hot-step speed: about 5.5 s/iter at 4 * 4096 tokens = about 3.0k tok/s
```

The typed local GB10 profile now renders `CPPMEGA_MTP_CE_KERNEL=cce`. Focused
standalone proof is in `tools/probes/linear_ce_probe.py`: on GB10, the
shared-weight 3-call CCE `reduction="sum"` pattern was finite with loss rel
error `9.78e-08`, and the synthetic BF16 CE timing favored CCE over Liger
(`118 ms` exact CCE sum-masked vs `380 ms` Liger mean-broadcast on
`[4096,3584] x [65536,3584]`).

The current local MXFP8 TE-transpose Linear backward receipt:

```text
log: /home/dave/logs/perf_mxfp8_te_emit_savedtranspose2_train_20260427_001248.log
profile: local_gb10_quarter, real clang semantic 4k data
steps: 4 train + 1 validation + 1 test
MTP: mtp_1 and mtp_2 losses logged, NaN iterations: 0
lm loss: 11.65989 -> 7.332075 over 4 train steps
validation loss: 7.147747
test loss: 6.961203
hot-step time: 5890.4 ms, 5894.2 ms for steps 3/4
hot-step speed: about 2.78k tokens/s at 4 * 4096 tokens
max allocated after step 2: 31871.13 MB
MXFP8 stats:
  dgrad=176, wgrad=176
  saved_transpose_operand=48
  copy_transpose=0
  bf16_fallback_dgrad=0, bf16_fallback_wgrad=0
  sidecar_registry_size=0
```

The newer FlashInfer/CUTLASS dense-GEMM backend keeps the same TE
transpose-emission contract but replaces the slow direct compact-loader path.
This is not the attention backend; GB10 attention is already routed through the
patched FA4 source tree by `attention_backend=flash`.

```text
backend: CPPMEGA_TE_MXFP8_BWD_BACKEND=flashinfer_cutlass
TE owns payload: rowwise MXFP8 uint8 storage
producer: compact rowwise scale -> FlashInfer/CUTLASS layout_128x4 uint8 scale
payload handoff: zero-copy uint8.view(torch.float8_e4m3fn)
fprop: x rowwise payload + weight rowwise payload.T
dgrad: dy rowwise payload + weight.T saved/on-demand payload.T
wgrad: dy.T saved/on-demand payload + x.T saved/on-demand payload.T
BF16 fallback: disabled
copy_transpose: zero in the accepted isolated Linear probe; nonzero in the
                full model where the adapter still bridges compact columnwise
                storage on demand
```

Latest full-model zero-sidecar receipt:

```text
log: /home/dave/logs/gb10_mxfp8_zero_sidecars_20260428_171130.log
profile: local_gb10_quarter, real clang semantic 4k data
steps: 6 train + validation + test
MTP: mtp_1 and mtp_2 losses logged, NaN iterations: 0
lm loss: 11.66119 -> 6.631005 over 6 train steps
validation loss: 5.908483
test loss: 5.862091
hot-step time: 5894.1, 5850.2, 5971.4, 5783.1 ms for steps 3-6
hot-step speed: about 2.74k-2.83k tokens/s at 4 * 4096 tokens
max allocated after step 2: 26468.95 MB
MXFP8 stats:
  mxfp8_tn_adapter_dgrad=60, mxfp8_tn_adapter_wgrad=60
  mxfp8_flashinfer_dgrad=204, mxfp8_flashinfer_wgrad=204
  mxfp8_flashinfer_fprop=0
  bf16_fallback_dgrad=0, bf16_fallback_wgrad=0
  native_passthrough_dgrad=0, native_passthrough_wgrad=0
  fallback_reasons={}
  mxfp8_tn_sidecar_attr_attached=0
  mxfp8_tn_sidecar_attr_attached_bytes=0
  mxfp8_tn_sidecar_registry_size=0
  mxfp8_tn_sidecar_registry_persistent=0
  mxfp8_tn_sidecar_registry_peak=0
  mxfp8_tn_sidecar_registry_current_bytes=0
  mxfp8_tn_sidecar_registry_peak_bytes=0
  mxfp8_tn_sidecar_tracked_attr_current_bytes=0
  mxfp8_tn_sidecar_tracked_attr_peak_bytes=0
  mxfp8_tn_adapter_te_emit=0
  mxfp8_tn_adapter_te_emit_deferred=4850
  mxfp8_tn_adapter_saved_transpose_operand=408
  mxfp8_tn_adapter_copy_transpose=3084
  mxfp8_tn_adapter_missing_sidecar_copy=3084
  mxfp8_norm_quantize_sidecar_bridge=100
```

This receipt proves zero persistent sidecars for the covered full-model run. It
does not prove zero-copy MXFP8 backward: the saved-transpose and copy-transpose
counters above show where the current bridge still materializes rowwise
transposed operands.

Latest grouped-direct full-model smoke:

```text
log: /home/dave/logs/gb10_mxfp8_grouped_direct_smoke9_20260428_183814.log
profile: local_gb10_quarter, real clang semantic 4k data
steps: 1 train + validation + test
lm loss: 11.66119
mtp_1 / mtp_2 loss: 11.97334 / 11.96537
validation / test loss: 10.66364 / 10.71262
max allocated / reserved: 24050.16 MB / 24866 MB
MXFP8 stats:
  mxfp8_flashinfer_dgrad=34, mxfp8_flashinfer_wgrad=34
  mxfp8_grouped_direct_dgrad=10, mxfp8_grouped_direct_wgrad=10
  mxfp8_grouped_direct_miss_dgrad=0, mxfp8_grouped_direct_miss_wgrad=0
  mxfp8_grouped_transpose_copy_fallback_dgrad=0
  mxfp8_grouped_transpose_copy_fallback_wgrad=0
  mxfp8_tn_adapter_copy_transpose=34
  mxfp8_tn_adapter_missing_sidecar_copy=34
  bf16_fallback_dgrad=0, bf16_fallback_wgrad=0
  native_passthrough_dgrad=0, native_passthrough_wgrad=0
  fallback_reasons={}
```

This receipt proves the grouped MoE dgrad/wgrad bridge no longer calls
`_cppmega_mxfp8_colwise_as_rowwise_transpose` and no longer materializes
per-expert transpose operands. The remaining copies are dense Linear copies.
The dense compact-columnwise path is wired behind
`--mxfp8-compact-columnwise-backward`; it is opt-in because the current dense
direct SM120 loader was correct but too slow on the full-model smoke.

## Next MXFP8 Acceptance Contract

The next accepted grouped direct backend and dense compact-columnwise path must
clear this counter contract on real `local_gb10_quarter` MXFP8 logs:

```text
dense direct backend used:
  mxfp8_cutlass_native_dgrad>0
  mxfp8_cutlass_native_wgrad>0

grouped direct backend used:
  mxfp8_grouped_direct_dgrad>0
  mxfp8_grouped_direct_wgrad>0
  mxfp8_grouped_direct_miss_dgrad=0
  mxfp8_grouped_direct_miss_wgrad=0
  mxfp8_grouped_transpose_copy_fallback_dgrad=0
  mxfp8_grouped_transpose_copy_fallback_wgrad=0

no transpose/copy materialization:
  zero calls to _cppmega_mxfp8_colwise_as_rowwise_transpose on accepted direct paths
  no x.T, dy.T, or weight.T MXFP8 tensor materialization
  mxfp8_tn_adapter_te_emit=0
  mxfp8_tn_adapter_te_emit_deferred=0
  mxfp8_tn_adapter_saved_transpose_operand=0
  mxfp8_tn_adapter_te_emit_swizzled=0
  mxfp8_tn_adapter_te_emit_swizzled_unavailable=0
  mxfp8_tn_adapter_copy_transpose=0
  mxfp8_tn_adapter_missing_sidecar_copy=0
  mxfp8_norm_quantize_sidecar_bridge=0

no persistent sidecars:
  mxfp8_tn_sidecar_attr_attached=0
  mxfp8_tn_sidecar_attr_attached_bytes=0
  mxfp8_tn_sidecar_registry_size=0
  mxfp8_tn_sidecar_registry_persistent=0
  mxfp8_tn_sidecar_registry_peak=0
  mxfp8_tn_sidecar_registry_current_bytes=0
  mxfp8_tn_sidecar_registry_peak_bytes=0
  mxfp8_tn_sidecar_tracked_attr_current_bytes=0
  mxfp8_tn_sidecar_tracked_attr_peak_bytes=0

no fallback/passthrough:
  bf16_fallback_dgrad=0
  bf16_fallback_wgrad=0
  native_passthrough_dgrad=0
  native_passthrough_wgrad=0
  fallback_reasons={}
```

Grouped direct is accepted only if one TE grouped GEMM call remains one grouped
backend launch. A Python wrapper that loops experts, calls
`_cppmega_mxfp8_colwise_as_rowwise_transpose`, or builds per-expert `x.T`,
`dy.T`, or `weight.T` operands is not the accepted grouped direct path even if
the numerical loss is finite.

## 2026-04-28 GB10 MXFP8 Profiling And Speed A/B

Current baseline command family:

```text
bash scripts/local_gb10_quarter_train.sh --fp8-recipe mxfp8 --muon-num-ns-steps 3
```

Artifacts:

```text
torch+memory profile:
  /home/dave/logs/gb10_mxfp8_teemit_torchmem_20260428_054124.log
  /home/dave/logs/gb10_mxfp8_teemit_torchmem_20260428_054124_torch_profile/train_step_2_cuda_table.txt

nsys full capture:
  /home/dave/logs/gb10_mxfp8_ns3_nsys_20260428_055404_nsys.nsys-rep
  /home/dave/logs/gb10_mxfp8_ns3_nsys_20260428_055404_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv

nsys delayed hot capture:
  /home/dave/logs/gb10_mxfp8_ns3_nsys_delay_exit_20260428_060936_nsys.nsys-rep
  /home/dave/logs/gb10_mxfp8_ns3_nsys_delay_exit_20260428_060936_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv
```

The delayed `nsys` launcher now supports `--nsys-capture-mode delay` with
`CPPMEGA_NSYS_DURATION=0`, meaning "start after delay and collect until normal
process exit".  This avoids the previous `nsys --duration` SIGTERM tail where
the report was generated but `torch.distributed.run` reported exit code 143.

End-to-end 8-step A/B on real clang 4k data, 16,384 tokens/step:

```text
variant                     hot avg ms   tok/s   iter loss   val loss   test loss
NS=3 correctness baseline      5653.4  2898.1    5.691503   5.144637    5.101255
NS=2 speed experiment          5354.5  3059.8    6.044881   5.475146    5.426576
```

`NS=2` is about 5.3% faster, but the short-run loss regression is too large for
the default correctness lane.  Keep `local_gb10_quarter` at `muon_num_ns_steps=3`;
use `--muon-num-ns-steps 2` only as an explicit risky speed experiment.

Other measured knobs:

```text
variant                         hot avg ms   tok/s    result
MTP CE = CCE baseline              5641.1   2904.4   keep
MTP CE = Liger                     5728.7   2860.0   slower
Mamba no-conv chunk size = 512      5686.3   2881.3   slower
Mamba no-conv chunk size = 128      5663.8   2892.7   slower and worse short loss
```

Hot `nsys` kernel ranking after warmup:

```text
_cce_backward_kernel                         ~17.8% to 19.7%
FlashInfer/CUTLASS SM120 MXFP8 GEMM          ~13.6% to 15.1%
BF16/CUTLASS GELU GEMM                       ~6% to 7%
_cce_lse_forward_kernel                      ~5% to 6%
TE mxfp8_scaling_transpose_cast              ~3.2% to 3.5%
TE MXFP8 quantize_kernel                     ~3.1% to 3.2%
FA4 attention backward                       ~1.5% to 1.7%
```

Immediate conclusion: forced FA4 is working and is not the current speed
bottleneck.  The next real speed work is kernel work, not launcher knobs:
reduce the 3-call CCE cost for main+2 MTP heads, reduce FlashInfer/CUTLASS
MXFP8 GEMM overhead for the current shapes, and remove TE transpose/quantize
launches by pushing GEMM-ready saved operands deeper into TE Linear/autograd.

Follow-up worktree integrations from the same hotspot list:

```text
CCE main+2 MTP launch fusion:
  status: implemented as RuntimePatchProfile.cce_fuse_main_mtp_ce / --cce-fuse-main-mtp-ce
  default: off
  correctness tests: scalar and per-token MTPLossAutoScaler parity
  real A/B logs:
    off: /home/dave/logs/gb10_cce_mtpfusion_off_20260428_155707.log
    on:  /home/dave/logs/gb10_cce_mtpfusion_on_20260428_155932.log
  hot iters 3-6:
    off 5671.65 ms, 2888.75 tok/s, max allocated 28461 MB
    on  5795.33 ms, 2827.11 tok/s, max allocated 28463 MB
  conclusion: finite and numerically close, but slower on this short GB10 run;
              keep it opt-in until the CCE kernel path itself is improved.

TE saved-operand transpose deferral:
  base:    /home/dave/logs/ab_te_linear_saved_operands_base_20260428_1603.log
  patched: /home/dave/logs/ab_te_linear_saved_operands_patched_20260428_1608.log
  max allocated after step 2: 28464.08 MB -> 28186.57 MB
  sidecar peak bytes:        3623522304 -> 3553576960
  hot iters 3-4:             5647.4 ms -> 5673.9 ms
  conclusion: useful memory-direction hook, not a speed win yet.

FlashInfer MXFP8 runner probe:
  tool: tools/probes/flashinfer_mxfp8_gemm_shape_probe.py
  control path: mm_mxfp8
  probe path:   direct_tactic through explicit FlashinferMxfp8RunnerConfig
  256x256x256:           0.0482 ms -> 0.0175 ms
  16384x3584x3584:       2.0938 ms -> 2.1772 ms
  16384x18944x3584:     44.8561 ms -> 44.5119 ms
  16384x3584x18944:     27.8601 ms -> 27.8218 ms
  conclusion: direct_tactic helps tiny probes and is near parity on big model
              shapes; default remains mm_mxfp8 until e2e nsys proves a win.
  e2e direct_tactic log:
    /home/dave/logs/gb10_mxfp8_direct_tactic_e2e_20260428_161541.log
  e2e vs mm_mxfp8 short control:
    direct_tactic hot avg 5641.2 ms, 2904.3 tok/s, max 28.459 GiB
    mm_mxfp8      hot avg 5671.6 ms, 2888.8 tok/s, max 27.794 GiB
  e2e conclusion: +0.54% tok/s is not enough to pay for the higher peak and
                  worse 6-step short loss; keep direct_tactic probe-only.
```

Speed comparison harness:

```bash
python tools/scripts/speed_compare.py \
  --run ns2=/home/dave/logs/gb10_mxfp8_ns2_speed_20260428_054440.log \
  --run ns3=/home/dave/logs/gb10_mxfp8_ns3_speed_20260428_054722.log \
  --baseline ns2 \
  --run-profile local_gb10_quarter \
  --fp8-recipe mxfp8 \
  --param-storage mxfp8 \
  --mxfp8-bwd-backend flashinfer_cutlass \
  --mxfp8-transpose-emit-backend te \
  --mxfp8-transpose-emit-swizzled \
  --mxfp8-transpose-emit-strict \
  --hot-step-start 3 \
  --nsys-kernel-csv ns3=/home/dave/logs/gb10_mxfp8_ns3_nsys_delay_exit_20260428_060936_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv \
  --nsys-top-n 5
```

`tools/scripts/speed_compare.py` reuses
`tools/profiling/compare_bf16_mxfp8.py` for train-log parsing and adds only
multi-run deltas plus optional `cuda_gpu_kern_sum` parsing.  The `nsys` status
is evidence metadata (`not_requested`, `csv_missing`, `no_kernel_rows`, `ok`),
not a training pass/fail gate.
