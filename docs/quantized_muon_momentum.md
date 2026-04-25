# Quantized Muon Momentum

Status date: 2026-04-25.

This note documents the current cppmega prototype for memory-reduced Muon
momentum. The goal is to replace the persistent BF16 Muon momentum tensor with
blockwise 8-bit state while still feeding the existing BF16 low-memory
Newton-Schulz path.

## What Exists

Implementation:

- `cppmega/megatron/quantized_muon_momentum.py`
- `cppmega/megatron/cuda_ext/quantized_muon_momentum.cpp`
- `cppmega/megatron/cuda_ext/quantized_muon_momentum.cu`

Tests and benchmark:

- `tests/test_quantized_muon_momentum.py`
- `scripts/bench_quantized_muon_momentum.py`

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

The important property is that the BF16 grad buffer is reused as the
Newton-Schulz input. There is no separate persistent BF16 momentum tensor and
no separate BF16 scratch allocation in the intended path.

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

## Current GB10 Measurements

Device: NVIDIA GB10, CUDA capability `(12, 1)`.

`4096 x 4096`, int8 state:

```text
Triton separate scratch update:     0.4548 ms
CUDA multi in-place update:         0.4626 ms
CUDA update + fused sumsq + norm:   0.7704 ms
CUDA update + torch norm/div:       0.8646 ms
BF16 momentum.lerp_:                0.4266 ms
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

- This is not yet wired into Megatron `TensorParallelMuon.step`.
- Current state_dict/checkpoint integration is not implemented.
- QKV split normalization must be handled per Q/K/V slice, not over the fused
  QKV tensor as a whole.
- Stochastic rounding is not implemented yet. The current kernel uses nearest
  rounding to keep parity debugging simple.
- The CUDA kernel handles contiguous tensors only.
- The update-and-normalize helper computes one global norm over the provided
  tensor list. Megatron QKV split integration must call it per slice or add a
  grouped/sliced normalization mode.

## Next Integration Target

Replace the current Megatron/Emerging path:

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
