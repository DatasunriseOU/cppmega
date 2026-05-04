# Mamba3 Mono WGMMA Copy Path Wave10 CuTe Integration - 2026-04-30

Status: active
Canonical: none
Date: 2026-04-30
Scope: Integration-ready contract for the 12-tile `uint4` vector copy path in the CuTe lane.

## Decision

Final recommendation: `integrate_vector_path_next`.

Vector path status: `True`.  TMA status:
`fallback_only_if_integrated_cute_lane_regresses_correctness_resources_or_timing`.

Reason: Wave9 correctness passed byte-for-byte against the scalar CUDA
reference, ptxas reported `40` registers and
zero spills, and the retained runtime measured `134.481597`
us vector copy vs `227.399206` us scalar copy on
`NVIDIA GB10`.

## Retained Artifacts

- Receipt: `docs/status/mamba3_mono_wgmma_copy_path_wave10_receipt_2026_04_30.json`
- Guard header consumed by the runtime probe: `tools/probes/mamba3_wgmma_wave10_copy_guards.hpp`
- Source runtime evidence: `docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json`
- Source ptxas evidence: `docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json`

## Guard Conditions

Compile-time guards:

- 64x64 BF16 tiles, `row_bytes=128`, `tile_bytes=8192`
- `uint4` lanes are 16 bytes and cover eight contiguous BF16 values
- no row or tile tail exists for this specialization
- exactly 10 global-source tiles and 2 CTA-local-stage tiles per chunk
- dynamic SMEM launch contract is `98304` bytes

Runtime guards:

- global, CTA-local, destination, and SMEM bases are 16-byte aligned
- row stride is 128 bytes and panel offsets are 16-byte aligned
- block size is a positive warp multiple
- device dynamic SMEM opt-in capacity is at least `98304` bytes
- a barrier follows the SMEM load and another barrier protects SMEM reuse

## Expected Data Layouts

| Tile group | Count | Source | Destination/layout |
| --- | ---: | --- | --- |
| K/Q/K_T/Q_T/state/dPhiO/PsiV | 10 | global | 64x64 BF16 physical rows, 128 bytes/row, 512 `uint4` lanes/tile |
| dstates panels | 2 | CTA-local stage | same 64x64 BF16 physical row contract |
| K_T/Q_T | 2 | global physical-layout/view operands | vector-compatible CuTe/GMMA layout, not per-column BF16 scatter |

## API Sketch

```c++
#include "tools/probes/mamba3_wgmma_wave10_copy_guards.hpp"

template <class GlobalTiles, class LocalStageTiles, class SmemTiles>
CUTE_DEVICE void copy_12tile_uint4_to_smem(
    GlobalTiles const& global_tiles,
    LocalStageTiles const& local_stage_tiles,
    SmemTiles& smem_tiles,
    int chunk_idx,
    int* status);
```

The generated code should call
`mamba3_wave10_copy::runtime_guard_bits(...)`, copy ten global tiles and two
CTA-local tiles as `uint4`, synchronize before WGMMA consumption, then
synchronize again before reusing the 12-tile staging buffer.

## Failure Modes

| Mode | Detection | Action |
| --- | --- | --- |
| misaligned source/destination | guard status bits | block vector launch; use scalar repair or TMA fallback |
| row or panel stride mismatch | generated layout guard | fix layout before enabling `uint4` |
| K_T/Q_T scalar scatter | transpose layout proof/codegen inspection | keep physical-layout view or switch to TMA smoke |
| masked tail emitted | guard contract/codegen inspection | remove tail path from this specialization |
| SMEM opt-in too small | dynamic SMEM guard | skip vector path on that target |
| resource regression | ptxas check | keep TMA fallback if regs spill or exceed budget |
| correctness regression | scalar byte-equality probe | block integration and investigate TMA fallback |

## Validation

```text
python tools/probes/mamba3_wgmma_wave10_copy_integration.py --runtime-result docs/status/mamba3_mono_wgmma_copy_path_wave9_runtime_2026_04_30.json --ptxas-ingest docs/status/mamba3_mono_wgmma_copy_path_wave9_ptxas_ingest_2026_04_30.json --check docs/status/mamba3_mono_wgmma_copy_path_wave10_receipt_2026_04_30.json --check-guard-header tools/probes/mamba3_wgmma_wave10_copy_guards.hpp --check-guide docs/status/mamba3_mono_wgmma_copy_path_wave10_cute_integration_2026_04_30.md
```

Focused tests:

```text
pytest tests/test_mamba3_wgmma_wave10_copy_integration.py tests/test_mamba3_wgmma_wave9_copy_probe.py
```
