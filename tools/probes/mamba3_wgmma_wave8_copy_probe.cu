#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"

namespace {

constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kBf16Bytes = 2;
constexpr int kVectorBytes = 16;
constexpr int kLogicalTileCount = 12;
constexpr int kGlobalTileCount = 10;
constexpr int kLocalStageTileCount = 2;
constexpr int kTileBytes = kTileRows * kTileCols * kBf16Bytes;
constexpr int kVectorsPerTile = kTileBytes / kVectorBytes;
constexpr int kTotalVectors = kLogicalTileCount * kVectorsPerTile;
constexpr int kDynamicSmemBytes = kLogicalTileCount * kTileBytes;

static_assert(kTileBytes == 8192);
static_assert(kVectorsPerTile == 512);
static_assert(kTileCols * kBf16Bytes == 128);
static_assert(kTotalVectors == 6144);
static_assert(kDynamicSmemBytes == 98304);
static_assert(kGlobalTileCount + kLocalStageTileCount == kLogicalTileCount);

using CuteTileShape = cute::Shape<cute::_64, cute::_64>;
static_assert(cute::size(CuteTileShape{}) == kTileRows * kTileCols);

}  // namespace

extern "C" __global__ void mamba3_wave8_narrow_copy_12tile_probe(
    const __nv_bfloat16* __restrict__ global_src,
    const __nv_bfloat16* __restrict__ local_stage_src,
    __nv_bfloat16* __restrict__ dst,
    int chunks) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];

  auto* smem_vectors = reinterpret_cast<uint4*>(smem_bytes);
  auto* global_vectors = reinterpret_cast<const uint4*>(global_src);
  auto* local_stage_vectors = reinterpret_cast<const uint4*>(local_stage_src);
  auto* dst_vectors = reinterpret_cast<uint4*>(dst);

  const uintptr_t alignment_bits = reinterpret_cast<uintptr_t>(global_src) |
                                   reinterpret_cast<uintptr_t>(local_stage_src) |
                                   reinterpret_cast<uintptr_t>(dst) |
                                   reinterpret_cast<uintptr_t>(smem_bytes);
  if ((alignment_bits & (kVectorBytes - 1)) != 0) {
    return;
  }

  for (int chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x) {
    const int chunk_dst_offset = chunk * kTotalVectors;
    const int chunk_global_offset = chunk * kGlobalTileCount * kVectorsPerTile;
    const int chunk_local_offset = chunk * kLocalStageTileCount * kVectorsPerTile;

    for (int tile = 0; tile < kLogicalTileCount; ++tile) {
      uint4* tile_smem = smem_vectors + tile * kVectorsPerTile;
      const uint4* tile_src = nullptr;
      int tile_src_offset = 0;
      if (tile < kGlobalTileCount) {
        tile_src = global_vectors;
        tile_src_offset = chunk_global_offset + tile * kVectorsPerTile;
      } else {
        tile_src = local_stage_vectors;
        tile_src_offset =
            chunk_local_offset + (tile - kGlobalTileCount) * kVectorsPerTile;
      }

      for (int vec = threadIdx.x; vec < kVectorsPerTile; vec += blockDim.x) {
        tile_smem[vec] = tile_src[tile_src_offset + vec];
      }
    }

    __syncthreads();

    for (int vec = threadIdx.x; vec < kTotalVectors; vec += blockDim.x) {
      dst_vectors[chunk_dst_offset + vec] = smem_vectors[vec];
    }
  }
}
