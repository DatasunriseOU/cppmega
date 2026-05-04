# Mamba3 Wave31 Lane C dPsiV_D H100 Helper Probe

Date: 2026-04-30

Branch: `worker/mamba3-wave31-dpsiv-helper-h100`

Source idea: Wave30 `worker/mamba3-wave30-helper-kernels`, narrowed to a real
H100 Modal compile/run evidence pass.

## Scope

H100 only. No H200 and no GCloud were used in this lane.

Candidate boundary:

```text
out_bf16 = bf16(dPsiV_fp32
                + dPhi_bf16 * D_fp32
                + gamma_fp32 * qk_dot_bf16^T @ dPhi_bf16)
```

The helper does not own full `bwd_bwd`, DK/DQ, DV, DPsi, state propagation, or
full production ownership/layout. It only probes whether this epilogue boundary
is compileable and resource-light on H100.

## Files

```text
tools/probes/mamba3_wave31_dpsiv_d_boundary_probe.py
scripts/modal_mamba3_wave31_dpsiv_helper_h100.py
artifacts/mamba3_wave31_dpsiv_helper_h100/wave31_dpsiv_helper_h100_20260430_r5/
```

## Commands

Local validation:

```bash
python -m py_compile scripts/modal_mamba3_wave31_dpsiv_helper_h100.py tools/probes/mamba3_wave31_dpsiv_d_boundary_probe.py
python tools/probes/mamba3_wave31_dpsiv_d_boundary_probe.py --tiles 16 --cs 8 --r 2 --p 16 --warmup 1 --iters 1 --json /tmp/mamba3_wave31_dpsiv_local_gate.json
```

H100 final run:

```bash
modal run scripts/modal_mamba3_wave31_dpsiv_helper_h100.py::launch_probe \
  --run-id wave31_dpsiv_helper_h100_20260430_r5 \
  --warmup 20 \
  --iters 100 \
  --timeout-s 1800
```

Artifact fetch:

```bash
modal volume get cppmega-mamba3-benchmarks \
  /benchmarks/mamba3_wave31_dpsiv_helper_h100/wave31_dpsiv_helper_h100_20260430_r5 \
  artifacts/mamba3_wave31_dpsiv_helper_h100/
```

## Modal Apps Started By This Lane

```text
ap-A3OE4tJcx8ldiqbvZiiXfM  cppmega-wave31-dpsiv-helper-h100  stopped  0 tasks
ap-Au1ccRzOkpENX4npFGEoDK  cppmega-wave31-dpsiv-helper-h100  stopped  0 tasks
ap-2XG0julRr8olbnGjpftWBO  cppmega-wave31-dpsiv-helper-h100  stopped  0 tasks
ap-AOkVsgLmKCUtze63659RK7  cppmega-wave31-dpsiv-helper-h100  stopped  0 tasks
ap-rZRFfyp5vDrBp5huQBmpOp  cppmega-wave31-dpsiv-helper-h100  stopped  0 tasks
```

The first three runs were harness/probe fixes:

```text
r1: harness JSON fallback parser crashed before report commit.
r2: torch load_inline generated a duplicate PYBIND11 module wrapper.
r3: compile succeeded, ptxas captured, exact correctness failed at max_abs=0.03125.
```

Runs `r4` and `r5` passed after using an explicit R-loop reference, disabling
TF32 in the reference path, removing `--use_fast_math`, and reporting BF16
one-ULP tolerance.

Two unrelated apps were still live during the final check and were left alone:

```text
ap-FwYHGBOJr9X9ID5r2fVQ53  cppmega-wave31-g8-reachability  ephemeral  1 task
ap-WfFnQaBpvi0LEqZw5yu29f  cppmega-wave31-lane-b-h100      ephemeral  1 task
```

## Final H100 Result

Run id: `wave31_dpsiv_helper_h100_20260430_r5`

GPU:

```text
NVIDIA H100 80GB HBM3, compute capability 9.0, 81559 MiB
CUDA 13.2, nvcc 13.2.78
torch 2.13.0.dev20260426+cu132
```

Shape:

```text
tiles=512, cs=128, r=2, p=128
```

Correctness vs torch BF16 reference:

```text
max_abs = 0.03125
mismatch_count = 443
atol = 0.03125
within_tolerance = true
exact = false
```

Timing:

```text
kernel = 0.120425596 ms
torch_reference = 0.732195511 ms
effective_bandwidth = 1046.094572 GiB/s
```

Memory:

```text
allocated_inputs_outputs = 135266304 bytes
after_bench_allocated = 235929600 bytes
peak_allocated_delta = 0 bytes
global_scratch = 0 bytes
```

Theoretical traffic:

```text
read_dpsiv_fp32 = 67108864 bytes
read_dphi_bf16 = 33554432 bytes
read_d_fp32 = 262144 bytes
read_qk_dot_bf16 = 524288 bytes
read_gamma_fp32 = 262144 bytes
write_out_bf16 = 33554432 bytes
global_scratch = 0 bytes
```

ptxas/resource evidence:

```text
dpsiv_d_boundary_kernel:
  registers = 32
  barriers = 0
  stack_frame = 0 bytes
  spill_stores = 0 bytes
  spill_loads = 0 bytes

cub EmptyKernel:
  registers = 4
  barriers = 0
  stack_frame = 0 bytes
  spill_stores = 0 bytes
  spill_loads = 0 bytes
```

## Judgment

This is not a production promotion yet.

Positive evidence:

```text
H100 compile: OK
resource pressure: low
spills: 0
global scratch: 0
standalone timing: 0.120 ms for the probe shape
```

Blocking evidence:

```text
correctness is BF16-tolerant, not exact
the helper is standalone and does not prove a real bwd_bwd speedup
production integration must remove at least ~0.12 ms of existing bwd_bwd work
before the helper launch/boundary is worth paying for
```

Safe-for-main judgment: safe as a guarded R&D probe/harness and evidence log
only. Do not wire this helper into production `mamba_mimo_bwd_bwd` until a real
A/B integration shows net positive H100/H200 full-backward timing and no new
expanded temps.
