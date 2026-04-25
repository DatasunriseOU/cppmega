# Memory Dtype Audit - 2026-04-25

Scope: local single-GB10 NAM56R-quarter run driven by
`scripts/local_gb10_quarter_train.sh` defaults:

- `--bf16`, `--fp8-format hybrid`, `CPPMEGA_FP8_RECIPE=tensorwise`
- `CPPMEGA_OPTIMIZER=muon`
- `CPPMEGA_MUON_QUANTIZED_MOMENTUM=1`, `CPPMEGA_MUON_QUANTIZED_MOMENTUM_DTYPE=int8`
- `CPPMEGA_MUON_SCALAR_OPTIMIZER=adam8bit`
- `CPPMEGA_USE_BF16_NO_MASTER_EMERGING_OPTIMIZER=1`
- `CPPMEGA_USE_BF16_NO_MASTER_EMERGING_FALLBACK_OPTIMIZER=1`
- `CPPMEGA_GRAD_REDUCE_IN_BF16=1`
- `CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER=0`
- `CPPMEGA_LOCAL_DDP_DISABLE_CONTIGUOUS_GRAD_BUFFER=1`
- default spec `cppmega.megatron.nam56r_noconv_spec build_cppmega_nam56r_noconv_stack_spec`

I inspected the existing cppmega memory hooks, the GB10 memory/perf notes, the
quarter-run launcher, quantized Muon state code, nanochat's `memory_debug.py`,
`memory_estimator.py`, `adamw.py`, `muon.py`, and `fp8_optimizer.py`, then
checked the local Megatron optimizer/DDP allocation paths used by this run.

## Runtime Inspector

Added a guarded inspector:

```bash
cd /home/dave/source/cppmega-memory-audit-agent
ROOT=$PWD \
PYTHONPATH=$PWD/tools:${PYTHONPATH:-} \
CPPMEGA_MEMORY_DTYPE_AUDIT=1 \
CPPMEGA_MEMORY_DTYPE_AUDIT_STEPS=1 \
CPPMEGA_TRAIN_ITERS=1 \
CPPMEGA_MICRO_BATCH_SIZE=1 \
CPPMEGA_GLOBAL_BATCH_SIZE=1 \
scripts/local_gb10_quarter_train.sh
```

For the current dirty main checkout run, keep `ROOT=/home/dave/source/cppmega`
and only prepend this worktree's `tools/` directory to `PYTHONPATH`.

The hook is installed by `tools/sitecustomize.py` only when
`CPPMEGA_MEMORY_DTYPE_AUDIT=1`. It writes JSON to
`CPPMEGA_MEMORY_DTYPE_AUDIT_OUT` or `/home/dave/logs/${RUN_ID}_dtype_audit.json`
and prints `[dtype_audit]` summary rows after model/optimizer setup and after
the first train step.

## Exact Storage Map

| Area | Persistent storage in this run | FP32 still present? |
| --- | --- | --- |
| Model parameters | BF16 after Megatron `Float16Module(...).bfloat16()` wraps the model. TE FP8 training does not make normal parameter storage FP8 here. | No persistent FP32 model params in the default no-conv spec. |
| Optimizer main params | None. Muon and scalar fallback both use `Float16NoMasterOptimizer`, so there is no `param.main_param` and no `fp32_from_float16_groups` master copy. | No FP32 main-param storage. |
| Main grads / DDP grad buffer | No contiguous DDP grad buffer. `--local-ddp-disable-contiguous-grad-buffer` clears `main_grad` and uses per-parameter `.grad`. BF16 params get BF16 grads. | No persistent FP32 grad buffer. Grad norm/clip reductions still compute in FP32. |
| `exp_avg` / `exp_avg_sq` for full model | Not active. `--use-precision-aware-optimizer` is off, and this is a Muon run, not distributed Adam. | No full-model FP32 `exp_avg`/`exp_avg_sq`. |
| Muon matrix state | `quantized_momentum_buffer.data`: int8, same shape as each Muon matrix. `quantized_momentum_buffer.absmax`: FP32, one value per 256-element block. | Yes: FP32 absmax scale metadata only, not a full momentum tensor. |
| Muon matrix update scratch | Quantized update overwrites the BF16 grad tensor with the updated/normalized momentum. | FP32 transient group sums/inv norms only. |
| Scalar fallback params | Same BF16 model params, no FP32 master because fallback no-master is enabled. | No FP32 params/masters. |
| Scalar fallback state (`adam8bit`) | bitsandbytes `state1`/`state2`. Params with `numel >= 4096`: uint8 states plus FP32 `absmax1`/`absmax2` per 256-value block and two FP32 256-entry qmaps. Params with `numel < 4096`: full FP32 `state1` and `state2`. | Yes: small fallback tensors keep full FP32 Adam states; large fallback tensors keep FP32 scale/qmap metadata. |
| Mamba no-conv `A_log`, `dt_bias`, `D` | `A_log` and `D` are constructed as FP32/default FP32 and converted to BF16 by `Float16Module`; `dt_bias` is constructed in `config.params_dtype` (BF16 for this run). Forward casts `A_log.float()` and `dt_bias.float()` before the SSD call; `D` stays BF16 in the default `D_has_hdim=False` path. | Persistent storage: no. Transient `A` and `dt_bias` tensors: yes. |
| Mamba no-conv B/C params | `B_norm_weight`, `C_norm_weight`, `B_bias`, `C_bias` are constructed as default FP32, then converted to BF16 by `Float16Module`. B/C activations are explicitly cast back to input dtype before scan. | Persistent storage: no. |
| Author Mamba3 alternate spec | `scripts/cppmega_fp8_shim.py` restores FP32 for `Mamba3.dt_bias`, `D`, `B_bias`, `C_bias`, `mimo_x`, `mimo_z`, `mimo_o` only for modules whose class name is exactly `Mamba3`. | Not active for the default no-conv spec; active only if running an Author Mamba3 spec. |
| M2RNN R-layer params | `A_log`/`dt_bias` are constructed FP32 and then converted to BF16 by `Float16Module`; other params use `config.params_dtype` BF16. | Persistent storage: no. |
| M2RNN saved/checkpoint tensors | Forward stores sparse recurrent checkpoints as FP32. Backward uses FP32 `dh_carry`, `dW_slabs`, and `y_chunk`. | Yes: FP32 activation/checkpoint and backward accumulator storage. |
| Sparse MLA / DSA FP8 kernels | Q/KV payloads are FP8/BF16 depending path; scale tensors, LSE, delta, and TileLang fragments named `acc_*` use `accum_dtype=T.float32`. | Yes: FP32 scale/output-reduction metadata and kernel accumulators. |
| TE dense GEMMs | Weight/activation storage is not FP32, but GEMM accumulation is high precision internally. | FP32/TF32-style accumulator behavior, not persistent tensor storage. |

## Where FP32 Actually Remains

Persistent FP32 storage in the default GB10 quarter Muon run is limited to:

1. Quantized Muon per-block `absmax` vectors.
2. bitsandbytes Adam8bit FP32 metadata for large scalar-fallback params:
   `absmax1`, `absmax2`, `qmap1`, `qmap2`.
3. bitsandbytes Adam8bit full FP32 `state1` and `state2` for scalar-fallback
   params below `min_8bit_size=4096`.
4. M2RNN FP32 recurrent checkpoints and backward accumulators.
5. Sparse MLA/DSA FP32 scale/LSE/delta tensors and TileLang accumulator
   fragments.

Transient FP32 compute remains in:

- Mamba no-conv `A = -exp(A_log.float())` and `dt_bias.float()` calls.
- Muon normalization/group-sums and grad-norm/clip reductions.
- TE/Triton/TileLang GEMM or scan accumulators.

Not present in this default run:

- FP32 optimizer main params for BF16 model params.
- FP32 persistent DDP grad buffer.
- Full-model FP32 `exp_avg`/`exp_avg_sq`.
- Persistent FP32 Mamba no-conv A/D/dt/B/C parameter storage.

If `CPPMEGA_USE_PRECISION_AWARE_OPTIMIZER=1` is used on an Adam run instead,
the patched Megatron config forces `main_grads=bf16`, `main_params=fp16`, and
`exp_avg/exp_avg_sq=uint8`, but that is a different optimizer path than the
current Muon quarter run.
