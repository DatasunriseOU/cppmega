# Cppmega Run Profiles And Token Flow

Status: canonical
Last updated: 2026-04-27
Scope: typed run-profile contract, local GB10 token flow, and precision/layout
boundaries for the current NAM56R-quarter debug lane.

## Run-Profile Contract

Core launch choices live in `cppmega/recipes/run_profiles.py`.

Each profile is built by a setter function that fills a `RunProfile` dataclass:

- `set_local_gb10_quarter_profile()`:
  single-GB10 correctness/profiling lane, 13 layers at NAM56R width,
  real 4k clang data, MTP depth 2, Liger MTP CE, tensorwise FP8, Flash
  Attention, no-master Muon, q8 Muon momentum, and local DDP contiguous grad
  buffer disabled.
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
  full attention: Flash Attention when the layer is not DSA
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
  CE route: Liger fused linear CE for local GB10 profile
  logged loss: mtp_1 loss

MTP predictor 2
  hidden in: BF16 [B, S, H]
  same local contract as predictor 1
  CE route: Liger fused linear CE
  logged loss: mtp_2 loss

main output head / CE
  hidden: BF16 [B, S, H]
  output weight: BF16 [V=65536, H=3584]
  local GB10 main head route:
    native LCE rejects cc=12.1, so Apple CCE/Liger-compatible fallback is used
  logits:
    fused CE avoids materializing full [B*S, V] logits as a persistent tensor
  reductions:
    loss reductions use FP32 internally where the fused CE kernel requires it
```

## Precision Boundaries

Persistent storage in the local profile:

- model parameters: BF16 after Megatron `Float16Module`
- ordinary hidden activations: BF16 between blocks
- optimizer main params: no FP32 master in the local no-master Muon lane
- Muon momentum: q8 data plus block absmax metadata
- fallback optimizer state: low-memory state selected by the profile

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
Linear input plus an MXFP8 transpose sidecar for the covered non-FSDP case.
Local TE `Linear` now prefers a GEMM-ready MXFP8 rowwise-transposed saved input
operand when the quantizer emitted one during forward. The original compact
columnwise activation is not kept for that Linear backward edge, and the shim
unregisters the transient sidecar so long runs do not accumulate stale registry
entries. This currently applies only when no backward input all-gather, CPU
offload, or FSDP scatter is involved.

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

- keep TE as the MXFP8 payload owner
- use TE transpose emit or cached rowwise-transpose sidecars for backward
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

The newer FlashInfer/CUTLASS backend keeps the same TE transpose-emission
contract but replaces the slow direct compact-loader path:

```text
backend: CPPMEGA_TE_MXFP8_BWD_BACKEND=flashinfer_cutlass
TE owns payload: rowwise MXFP8 uint8 storage
producer: compact rowwise scale -> FlashInfer/CUTLASS layout_128x4 uint8 scale
payload handoff: zero-copy uint8.view(torch.float8_e4m3fn)
fprop: x rowwise payload + weight rowwise payload.T
dgrad: dy rowwise payload + weight.T sidecar payload.T
wgrad: dy.T sidecar payload + x.T sidecar payload.T
BF16 fallback: disabled
copy_transpose: zero in the accepted shim probe
```
