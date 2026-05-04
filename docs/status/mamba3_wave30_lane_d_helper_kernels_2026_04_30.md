# Mamba3 Wave30 Lane D Helper Kernel Probe

Date: 2026-04-30

Branch: `worker/mamba3-wave30-helper-kernels`

## Decision

Candidate selected: `dPsiV_D` bf16 cast/fuse boundary.

This is intentionally not another monolithic `bwd_bwd` attempt. The helper
starts after the large GEMM accumulation has already produced `dPsiV_fp32` and
only fuses the local epilogue terms that currently create a precision boundary:

```text
out_bf16 = bf16(dPsiV_fp32
                + dPhi_bf16 * D_fp32
                + gamma_fp32 * qk_dot_bf16^T @ dPhi_bf16)
```

## Interface

Prototype probe:

```text
tools/probes/mamba3_wave30_dpsiv_d_boundary_probe.py
```

Inputs:

```text
dpsiv_fp32[tiles, cs*r, p]
dphi_bf16[tiles, cs*r, p]
d_fp32[tiles, p]
qk_dot_bf16[tiles, cs, r, r]
gamma_fp32[tiles, cs]
```

Output:

```text
out_bf16[tiles, cs*r, p]
```

Scratch:

```text
global scratch: 0 bytes
register scratch: one R-loop accumulator per output element
```

The interface is small enough to serve either a TileLang epilogue split or a
CuTe helper boundary. It does not own the large GEMMs, loop-carried `dstates`,
DV, DPsi, or dq/dk paths.

## Local Resource Gate

Wave30 Lane D policy for this probe is H100-local only; H200 is explicitly
forbidden. The local machine is not an H100:

```text
$ nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader
NVIDIA GB10, 12.1, [N/A]
```

The probe therefore refused to build or run:

```json
{
  "candidate": "dpsiv_d_bf16_boundary",
  "compiled": false,
  "gate": {
    "capability": [
      12,
      1
    ],
    "name": "NVIDIA GB10",
    "ok": false,
    "reason": "requires_local_h100_sm90"
  },
  "host": "gx10-9cd4",
  "status": "NO_GO_RESOURCE_GATE"
}
```

Because the probe did not compile under the resource gate:

```text
ptxas/resources: not collected
timing: not collected
memory measurement: not collected
```

## H100 Command

Run only on a local H100 host:

```bash
python tools/probes/mamba3_wave30_dpsiv_d_boundary_probe.py \
  --tiles 512 --cs 128 --r 2 --p 128 \
  --warmup 20 --iters 100 --verbose-build
```

`--verbose-build` passes `-Xptxas=-v`, so the compile log should include
register, spill, constant-memory, and shared-memory resource lines. The JSON
result reports correctness, kernel timing, torch-reference timing, theoretical
traffic, peak allocated delta, and scratch bytes.

## Lane B/C Recommendation

Do not feed this helper into Lane B/C as an implementation dependency yet. It
has a clean interface and good blast radius, but it is currently only a
resource-gated prototype on this machine.

Feed the idea, not the code, into Lane B/C planning if they need a narrow
epilogue boundary that avoids expanded `D*dPhi` or `qk_dot @ dPhi` temporaries.
Promote the helper only after an H100 compile/run captures ptxas resources,
correctness, timing, and memory.
