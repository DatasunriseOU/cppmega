#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include "cute/tensor.hpp"

namespace {

constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kBf16Bytes = 2;
constexpr int kVectorBytes = 16;
constexpr int kTileBytes = kTileRows * kTileCols * kBf16Bytes;
constexpr int kVectorsPerTile = kTileBytes / kVectorBytes;

static_assert(kTileBytes == 8192);
static_assert(kVectorsPerTile == 512);
static_assert(kTileCols * kBf16Bytes == 128);

using CuteTileShape = cute::Shape<cute::_64, cute::_64>;
static_assert(cute::size(CuteTileShape{}) == kTileRows * kTileCols);

}  // namespace

extern "C" __global__ void mamba3_wave7_narrow_copy_probe(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int tiles) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];
  auto* smem_vectors = reinterpret_cast<uint4*>(smem_bytes);
  auto* src_vectors = reinterpret_cast<const uint4*>(src);
  auto* dst_vectors = reinterpret_cast<uint4*>(dst);

  for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
    const int tile_offset = tile * kVectorsPerTile;
    for (int vec = threadIdx.x; vec < kVectorsPerTile; vec += blockDim.x) {
      uint4 value = src_vectors[tile_offset + vec];
      smem_vectors[vec] = value;
    }
    __syncthreads();
    for (int vec = threadIdx.x; vec < kVectorsPerTile; vec += blockDim.x) {
      dst_vectors[tile_offset + vec] = smem_vectors[vec];
    }
  }
}
