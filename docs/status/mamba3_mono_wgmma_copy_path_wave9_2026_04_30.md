# Mamba3 Mono WGMMA Copy Path Wave9 - 2026-04-30

Status: pass for the standalone 12-tile `uint4` runtime probe.

Canonical receipt:
`docs/status/mamba3_mono_wgmma_copy_path_wave9_receipt_2026_04_30.json`

Retained evidence:

- `docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json`
- `docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json`
- `tools/probes/mamba3_wgmma_wave9_copy_probe.cu`
- `tools/probes/mamba3_wgmma_wave9_copy_probe.py`
- `tests/test_mamba3_wgmma_wave9_copy_probe.py`

## What Changed

Wave9 moves beyond the Wave8 compile-only receipt.  The new standalone CUDA
probe runs both paths on GB10:

- vector path: `mamba3_wave9_uint4_copy_12tile_probe`
- scalar reference: `mamba3_wave9_scalar_copy_12tile_reference`

The vector kernel copies the same representative 12 logical 64x64 BF16 tiles
per chunk:

- 10 global-source tiles
- 2 CTA-local-stage tiles
- 6144 `uint4` lanes/chunk
- 98304 bytes/chunk
- 98304 bytes dynamic SMEM launch contract

Correctness is byte equality against the scalar CUDA kernel, including matching
FNV-1a checksums.  Timing is ingested as measured evidence only; no speedup
threshold is used as a correctness gate.

## Compile And Runtime Command

```text
python tools/probes/mamba3_wgmma_wave9_copy_probe.py \
  --compile-run \
  --chunks 128 \
  --warmup 5 \
  --iters 40 \
  --write-runtime docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json \
  --write-ptxas-ingest docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json \
  --write docs/status/mamba3_mono_wgmma_copy_path_wave9_receipt_2026_04_30.json
```

## Evidence

Runtime:

| field | value |
| --- | ---: |
| status | pass |
| chunks | 128 |
| logical payload bytes/iteration | 12582912 |
| copy-stage bytes/iteration | 25165824 |
| mismatched elements | 0 |
| scalar checksum | 7518092082649847683 |
| vector checksum | 7518092082649847683 |
| vector avg time | 134.481597 us |
| scalar avg time | 227.399206 us |
| vector copy-stage throughput | 174.280352 GiB/s |
| scalar payload throughput | 51.533821 GiB/s |
| vector/scalar time ratio | 1.690932 |

ptxas:

| field | value |
| --- | ---: |
| CUDA arch | sm_121 |
| registers/thread | 40 |
| static SMEM bytes | 16 |
| dynamic SMEM bytes | 98304 |
| total SMEM bytes | 98320 |
| spill stores bytes | 0 |
| spill loads bytes | 0 |

The retained receipt has no blockers.

## Validation

```text
python tools/probes/mamba3_wgmma_wave9_copy_probe.py \
  --runtime-result docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json \
  --ptxas-ingest docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json \
  --check docs/status/mamba3_mono_wgmma_copy_path_wave9_receipt_2026_04_30.json
```

Focused tests:

```text
pytest tests/test_mamba3_wgmma_wave9_copy_probe.py
```

## Wave10 Recommendation

Integrate the `uint4` vector copy into the CuTe lane first.  Carry forward the
Wave8 alignment/layout guards and the Wave9 runtime launch contract, including
the post-destination-write `__syncthreads()` before reusing the 12-tile SMEM
staging buffer across chunks.

Switch to TMA descriptor smoke only if the integrated CuTe lane regresses
correctness, spills, or timing.
