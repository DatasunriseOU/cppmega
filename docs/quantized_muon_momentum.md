# Quantized Muon Momentum

Status date: 2026-04-30.

This note documents the current cppmega prototype for memory-reduced Muon
momentum. The production-shaped optimizer path replaces the persistent BF16
Muon momentum tensor with blockwise 8-bit state while still feeding the
existing BF16 low-memory Newton-Schulz path. Wave18B adds a staged MXFP8 carrier
probe for removing that BF16 handoff, but it is not yet wired into full model
training.

## What Exists

Implementation:

- `cppmega/megatron/quantized_muon_momentum.py`
- `cppmega/megatron/cuda_ext/quantized_muon_momentum.cpp`
- `cppmega/megatron/cuda_ext/quantized_muon_momentum.cu`

Tests and benchmark:

- `tests/test_quantized_muon_momentum.py`
- `tests/test_muon_dtype_audit.py`
- `scripts/bench_quantized_muon_momentum.py`
- `tools/probes/qmuon_mxfp8_carrier_ns_probe.py`

The state format is:

- `state.data`: `torch.int8` or `torch.uint8`, same shape as the momentum tensor.
- `state.absmax`: `torch.float32`, one value per block of 256 elements.
- Quantization: symmetric nearest rounding with effective range `[-127, 127]`.
- Block size: 256 elements.

The CUDA extension provides two production-shaped update paths:

- `quantized_muon_momentum_update_multi_`: updates many tensors in one CUDA
  launch and overwrites each grad tensor with the updated Muon momentum.
- `quantized_muon_momentum_update_multi_and_normalize_`: same update, plus
  sum-of-squares collection in the update kernel, followed by in-place
  Frobenius normalization for Newton-Schulz input.
- `quantized_muon_momentum_update_multi_and_normalize_groups_`: grouped/sliced
  variant for fused QKV and sharded tensors. It accumulates one sumsq per
  logical norm group, optionally all-reduces those sums over a tensor-parallel
  process group, then scales every block by its group norm.
- `quantized_muon_momentum_update_mxfp8_carrier_`: Wave18B single-tensor probe
  path. It updates q8 state and emits a normalized rowwise MXFP8 carrier
  (`uint8` E4M3 payload plus `uint8` E8M0 inverse scales) without overwriting
  the grad tensor with BF16 scratch.

The important property is that the BF16 grad buffer is reused as the
Newton-Schulz input. There is no separate persistent BF16 momentum tensor and
no separate BF16 scratch allocation in the intended grouped path. The Wave18B
carrier probe is the first step toward removing this BF16 NS input, but the
Megatron optimizer still uses the grouped BF16 handoff today.

## Algorithm

For each 256-element block:

1. Load quantized old momentum and the block `absmax`.
2. Dequantize in FP32 registers.
3. Load BF16/FP16/FP32 grad.
4. Compute:

   ```text
   m = beta * old_m + (1 - beta) * grad
   ```

5. Store `m` back into the grad tensor.
6. Compute new block `absmax`.
7. Requantize `m` to int8/uint8.
8. Store quantized momentum and new block `absmax`.
9. Optionally accumulate block `sum(m * m)` for Newton-Schulz normalization.

The CUDA kernel uses a bitsandbytes-style element layout: 64 threads per block,
4 elements per thread, for 256 elements per quantization block.

## Wave18B MXFP8 Carrier Probe

The Wave18B branch adds a non-BF16 carrier building block:

```text
q8 state + grad -> q8 state + normalized MXFP8 carrier
```

Carrier layout:

- `carrier.rowwise_data`: `torch.uint8`, same 2D shape as the source tensor,
  storing E4M3 payload bytes.
- `carrier.rowwise_scale_inv`: `torch.uint8`, shape `(rows, ceil(cols / 32))`,
  storing E8M0 inverse scales.
- `carrier.inv_norm`: `torch.float32`, one scalar inverse Frobenius norm.

Locked GB10 microprobe:

```text
/home/dave/logs/wave18B_qmuon_mxfp8_carrier_probe_20260430_002.log
carrier_gram_rel_l2=0.00172372
carrier_finite=True
gram_finite=True
q_update_emit_plus_mxfp8_gram=2.0756 ms
bf16_update_norm_gram=3.8241 ms
speedup_vs_bf16=1.842x
```

Audit evidence:

```text
/home/dave/logs/wave18B_qmuon_mxfp8_carrier_audit_20260430_001.log
bf16_owned_path_observed=0
qmuon_mxfp8_carrier_update_calls=1
qmuon_carrier_rowwise_data_dtype_uint8_tensors=1
qmuon_carrier_rowwise_scale_dtype_uint8_tensors=1
qmuon_carrier_inv_norm_dtype_float32_tensors=1
qmuon_grad_dtype_float16_tensors=1
```

Current limit: this is a one-tensor microprobe, not a full optimizer route.
The existing CUTLASS MXFP8 helper still requires BF16 output for the Gram GEMM,
so the carrier storage is non-BF16 but the full Newton-Schulz computation is
not yet strict no-BF16.

See
`docs/status/qmuon_mxfp8_carrier_wave18B_2026_04_30.md`
for the staged `TensorParallelMuon.step` / `TensorParallelMuon.orthogonalize`
integration plan and acceptance gate.

## Grouped QKV And Sharding

Grouped normalization is driven by reusable metadata:

- `QuantizedMuonNormSegment(tensor_index, start, length, group_id)`
- `build_quantized_muon_norm_plan(states, segments, num_groups=...)`
- `QuantizedMuonNormPlan.block_group_ids`: one int64 group id per 256-value
  quantization block.

For fused QKV, build repeated segments with shared logical group ids:

- Q segments -> `group_id=0`
- K segments -> `group_id=1`
- V segments -> `group_id=2`

This matches current Megatron Muon semantics: Q, K, and V are normalized
independently, even when the storage tensor is fused/interleaved.

Shard contract:

- Momentum state is local to each shard: int8/uint8 data plus local absmax.
- Norm groups are logical and must use the same `group_id` order on every TP
  rank.
- Each rank builds a local `QuantizedMuonNormPlan` for its shard.
- `quantized_muon_momentum_update_multi_and_normalize_groups_(..., tp_group=...)`
  computes local group sums in the update kernel, all-reduces the `num_groups`
  FP32 sums across `tp_group`, then scales local grads by the global norm.
- Segment boundaries must align to 256-value quantization block boundaries,
  except a segment ending at the tensor's final partial block. This keeps the
  update kernel one-pass and avoids per-element segment lookup.

## Current GB10 Measurements

Device: NVIDIA GB10, CUDA capability `(12, 1)`.

`4096 x 4096`, int8 state:

```text
Triton separate scratch update:     0.4548 ms
CUDA multi in-place update:         0.4454 ms
CUDA update + fused sumsq + norm:   0.7543 ms
CUDA update + torch norm/div:       0.8477 ms
BF16 momentum.lerp_:                0.4192 ms
```

`65536 x 3584`, int8 state:

```text
Triton separate scratch update:     6.1071 ms
CUDA multi in-place update:         6.2631 ms
CUDA update + fused sumsq + norm:   10.3112 ms
CUDA update + torch norm/div:       11.8285 ms
BF16 momentum.lerp_:                5.9131 ms
```

Interpretation:

- Raw int8 update is still slightly slower than BF16 `lerp_`, because the
  kernel also performs per-block reduction, scaling, rounding, and requant.
- Persistent state memory drops from 2 bytes per parameter for BF16 momentum to
  about 1 byte per parameter plus a small FP32 scale vector.
- The update-and-normalize path is already faster than update followed by
  ordinary torch norm/div because the sum-of-squares work is fused into the
  momentum update pass.

Nsight Systems microbench capture:

```text
/home/dave/logs/qmuon_microbench_nsys.nsys-rep
```

Top relevant kernels for `4096 x 4096`, int8, 5 measured iterations:

```text
qmuon_update_multi_kernel:          21 launches, 9.304 ms total
_quantized_muon_momentum_update:    13 launches, 5.769 ms total
BF16 torch norm reduce:              7 launches, 1.089 ms total
```

The symbolized `qmuon_update_multi_kernel` is the CUDA multi-tensor in-place
path. The remaining `_quantized_muon_momentum_update_kernel` launches are from
the older single-tensor benchmark path, retained for parity comparison.

PyTorch profiler found that the initial extension wrapper was allocating and
copying five small metadata arrays on every multi-update call. That is now
packed into one metadata tensor and one H2D copy:

```text
before: aten::empty 50 calls, HtoD memcpy 50 calls per 10 updates
after:  aten::empty 10 calls, HtoD memcpy 10 calls per 10 updates
```

Post-patch profiler trace:

```text
/home/dave/logs/qmuon_torch_profiler_packed_meta
qmuon_update_multi_kernel: 10 launches, 4.430 ms total, 443 us avg
Memcpy HtoD metadata:      10 copies,   4.256 us total
```

Grouped QKV path, int8 state, synthetic interleaved QKV shape
`(4096, 3584)`, split `(2, 1, 1)`, 3072 segments, 57344 quantization blocks:

```text
fused_group_update_norm:      0.8950 ms
update_plus_torch_qkv_norm:   1.9239 ms
ratio:                        0.47x
```

PyTorch profiler, 10 grouped updates on the same shape:

```text
qmuon_update_multi_kernel:                  3.888 ms total, 388.8 us avg
qmuon_scale_multi_by_group_from_sumsq:      2.541 ms total, 254.1 us avg
benchmark DtoD grad copy:                   2.425 ms total, not part of real optimizer step
```

Nsight Compute report:

```text
/home/dave/logs/qmuon_qkv_grouped_ncu.ncu-rep
```

Key `ncu --set basic` metrics:

```text
qmuon_update_multi_kernel:
  duration avg:             393.65 us
  achieved occupancy avg:    80.69%
  registers/thread:          26
  memory throughput:         32.69%

qmuon_scale_multi_by_group_from_sumsq_kernel:
  duration avg:             254.77 us
  achieved occupancy avg:    91.97%
  registers/thread:          16
  memory throughput:         14.27%
```

`ncu` command used:

```bash
sudo -E /usr/local/cuda/bin/ncu \
  --target-processes all \
  --kernel-name-base demangled \
  --kernel-name 'regex:.*qmuon_(update_multi|scale_multi_by_group_from_sumsq)_kernel.*' \
  --launch-skip 0 \
  --launch-count 4 \
  --set basic \
  --force-overwrite \
  --export /home/dave/logs/qmuon_qkv_grouped_ncu \
  /home/dave/cppmega-venv/bin/python \
  scripts/bench_quantized_muon_momentum.py \
  --qkv-grouped --rows 4096 --cols 3584 --warmup 1 --iters 2 --storage int8
```

## Nsight Counter Permissions

Official NVIDIA references:

- <https://developer.nvidia.com/ERR_NVGPUCTRPERM>
- <https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html>

Current loaded driver state before reboot:

```text
/proc/driver/nvidia/params: RmProfilingAdminOnly: 1
```

Persistent Linux config installed:

```bash
sudo install -m 0644 /dev/stdin /etc/modprobe.d/nvidia-profiler-counters.conf <<'EOF'
# Allow non-root Nsight Compute/Nsight Systems/CUPTI access to NVIDIA GPU
# performance counters. Required to avoid ERR_NVGPUCTRPERM when profiling
# qmuon and other CUDA kernels as the normal development user.
#
# NVIDIA official guidance:
# https://developer.nvidia.com/ERR_NVGPUCTRPERM
options nvidia NVreg_RestrictProfilingToAdminUsers=0
EOF

sudo update-initramfs -u -k all
```

The config is present in the current initrd:

```bash
lsinitramfs /boot/initrd.img-$(uname -r) | grep nvidia-profiler-counters
```

A reboot or a safe NVIDIA module unload/reload is still required for
`RmProfilingAdminOnly` to become `0` in the loaded driver.

Immediate profiling path used for this session:

```bash
sudo setcap cap_sys_admin+ep \
  /usr/local/cuda-13.2/nsight-compute-2026.1.1/target/linux-desktop-t210-a64/ncu
```

Setting `cap_sys_admin` only on `ncu` was not enough for this GB10 userspace
path: `ncu` connected, but counter collection still failed. Running `ncu` with
`sudo -E` worked and produced `/home/dave/logs/qmuon_qkv_grouped_ncu.ncu-rep`.

## Correctness Coverage

Current focused test:

```bash
PYTHONPATH=. /home/dave/cppmega-venv/bin/python -m pytest -q tests/test_quantized_muon_momentum.py
```

Current result:

```text
8 passed
```

The tests cover:

- int8 and uint8 states.
- Single-tensor update versus dequantized reference.
- Repeated update drift versus BF16 reference.
- Reusing grad as scratch.
- CUDA multi-tensor in-place update.
- Update with per-block sumsq.
- Update plus in-place normalization for Newton-Schulz input.

## Known Limitations

- A local Megatron worktree integration exists in
  `/home/dave/megatron-lm/megatron/core/optimizer/emerging_optimizers.py`.
  It is intentionally not committed from this repository because the Megatron
  worktree is a detached, heavily dirty checkout with many unrelated local
  changes.
- Current state_dict/checkpoint integration is not implemented.
- Current Megatron integration preserves QKV correctness by reusing the
  existing `orthogonalize()` path after the quantized momentum update. That
  means Q/K/V are still split and normalized independently. The fused
  update-and-normalize helper is not used for QKV yet because it currently
  computes one global norm over its tensor list.
- Stochastic rounding is not implemented yet. The current kernel uses nearest
  rounding to keep parity debugging simple.
- The CUDA kernel handles contiguous tensors only.
- The update-and-normalize helper computes one global norm over the provided
  tensor list. Megatron QKV split integration must call it per slice or add a
  grouped/sliced normalization mode.

## Local Megatron Integration

The local Megatron patch replaces the current Emerging optimizer path:

```python
state["momentum_buffer"].lerp_(grad, 1 - group["momentum"])
```

with quantized state:

```python
q_state = state["quantized_momentum_buffer"]
quantized_muon_momentum_update_multi_(q_states, grads, beta=group["momentum"])
```

For the low-memory Newton-Schulz path, use the grad tensor itself as the BF16
input. For QKV parameters, split Q/K/V first and normalize each slice
independently before Newton-Schulz, matching current Muon behavior.

Smoke result on the local GB10 quarter NAM56R launcher:

```text
CPPMEGA_TRAIN_ITERS=1 CPPMEGA_MICRO_BATCH_SIZE=4 CPPMEGA_GLOBAL_BATCH_SIZE=4 \
  scripts/local_gb10_quarter_train.sh
```

Observed result:

```text
iteration:      1/1
lm loss:        1.165463E+01
mtp_1 loss:     1.164793E+01
grad norm:      ~76.95
skipped/nan:    0/0
max allocated:  27760.22 MB
max reserved:   29524.00 MB
```

Important profiling note: profiling the wrapper training script with `nsys`
did not capture the optimizer worker reliably yet. The extension microbench
does capture and symbolize the `qmuon_update_multi_kernel`; the full training
profile still needs either direct `torchrun` profiling or an explicit
Megatron/PyTorch profiler window around `optimizer.step()`.
