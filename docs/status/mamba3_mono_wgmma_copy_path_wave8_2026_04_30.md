# Mamba3 Mono WGMMA Copy Evidence Wave8 - 2026-04-30

Status: pass for the narrow 128-bit evidence gate.

Canonical receipt:
`docs/status/mamba3_mono_wgmma_copy_path_wave8_receipt_2026_04_30.json`

Retained evidence:

- `docs/status/mamba3_mono_wgmma_copy_path_wave8_alignment_2026_04_30.json`
- `docs/status/mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30.json`
- `tools/probes/mamba3_wgmma_wave8_copy_evidence.py`
- `tools/probes/mamba3_wgmma_wave8_copy_probe.cu`
- `tests/test_mamba3_wgmma_wave8_copy_evidence.py`

## What Changed

Wave8 promotes the Wave7 placeholder into an actual receipt for the
`narrow_vector_128b_safe_attempt` gate:

- all 12 Wave6 logical tile movements have per-tile alignment/layout evidence
- `K_T` and `Q_T` have explicit physical-layout transpose proofs
- ptxas metadata is retained as JSON and evaluated by the Wave6 resource gate
- a representative 12-logical-tile CUDA/CuTe `uint4` copy probe compiles on GB10

The receipt still does not claim full monolithic WGMMA timing. It proves the
narrow-copy evidence gate is complete enough to enter the next implementation
wave.

## Evidence Fields

Aggregate alignment/layout fields:

- `vector_bytes=16`
- `tile_rows=64`, `tile_cols=64`, `dtype_bytes=2`
- `row_bytes=128`, `tile_bytes=8192`
- `row_tail_bytes=0`, `tile_tail_bytes=0`
- `global_base_alignment_bytes=16`, `smem_base_alignment_bytes=16`
- `runtime_global_alignment_guard=true`
- `runtime_smem_alignment_guard=true`
- `row_stride_alignment_bytes=128`
- `uses_16b_contiguous_vector_type=true`
- `masked_tail_path_present=false`
- `tiles_covered`: all 12 Wave6 tiles
- `kt_qt_vector_compatible_layout=true`

Per-tile required fields additionally include source/destination spaces,
`tma_candidate`, vectors per row/tile, source/destination base alignment
requirements, runtime source/destination guards, panel-offset alignment,
`vector_type=uint4`, layout kind/proof, guard expressions, and
`transpose_layout_proof`.

The transpose proof for `K_T` and `Q_T` is intentionally narrow: they pass only
as GMMA physical-layout/view operands with 128-byte physical rows and no
per-column BF16 scatter.

## Compile Evidence

Command:

```text
python tools/probes/mamba3_wgmma_wave8_copy_evidence.py \
  --compile-probe \
  --write-ptxas-ingest docs/status/mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30.json
```

Captured fields:

| field | value |
| --- | ---: |
| kernel | `mamba3_wave8_narrow_copy_12tile_probe` |
| CUDA arch | `sm_121` |
| registers/thread | `40` |
| static SMEM bytes | `0` |
| dynamic SMEM bytes | `98304` |
| spill stores bytes | `0` |
| spill loads bytes | `0` |

The dynamic SMEM value is the probe launch contract:
`12 tiles * 8192 B/tile = 98304 B`.

## Validation

```text
python tools/probes/mamba3_wgmma_wave8_copy_evidence.py \
  --alignment-evidence docs/status/mamba3_mono_wgmma_copy_path_wave8_alignment_2026_04_30.json \
  --ptxas-ingest docs/status/mamba3_mono_wgmma_copy_path_wave8_ptxas_ingest_2026_04_30.json \
  --check docs/status/mamba3_mono_wgmma_copy_path_wave8_receipt_2026_04_30.json
```

Focused tests:

```text
pytest tests/test_mamba3_wgmma_wave8_copy_evidence.py
```

## Wave9 Recommendation

Proceed with the vector path first. Integrate the narrow `uint4` copy path into
the monolithic CuTe/WGMMA skeleton and carry over the exact ptr%16,
row-stride, panel-offset, and `K_T/Q_T` physical-layout guards. Keep TMA
descriptor smoke as the fallback branch if integration spills or timing is red.
