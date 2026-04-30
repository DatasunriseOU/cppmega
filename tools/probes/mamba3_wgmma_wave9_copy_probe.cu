#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <limits>
#include <vector>

#include "cute/tensor.hpp"

namespace {

constexpr int kTileRows = 64;
constexpr int kTileCols = 64;
constexpr int kBf16Bytes = 2;
constexpr int kVectorBytes = 16;
constexpr int kLogicalTileCount = 12;
constexpr int kGlobalTileCount = 10;
constexpr int kLocalStageTileCount = 2;
constexpr int kTileElements = kTileRows * kTileCols;
constexpr int kTileBytes = kTileElements * kBf16Bytes;
constexpr int kVectorsPerTile = kTileBytes / kVectorBytes;
constexpr int kElementsPerVector = kVectorBytes / kBf16Bytes;
constexpr int kTotalVectors = kLogicalTileCount * kVectorsPerTile;
constexpr int kTotalElements = kLogicalTileCount * kTileElements;
constexpr int kDynamicSmemBytes = kLogicalTileCount * kTileBytes;

static_assert(kTileBytes == 8192);
static_assert(kVectorsPerTile == 512);
static_assert(kTileCols * kBf16Bytes == 128);
static_assert(kElementsPerVector == 8);
static_assert(kTotalVectors == 6144);
static_assert(kDynamicSmemBytes == 98304);
static_assert(kGlobalTileCount + kLocalStageTileCount == kLogicalTileCount);

using CuteTileShape = cute::Shape<cute::_64, cute::_64>;
static_assert(cute::size(CuteTileShape{}) == kTileRows * kTileCols);

struct Options {
  int chunks = 128;
  int warmup_iterations = 5;
  int timed_iterations = 40;
  int block_threads = 256;
  int grid_blocks = 0;
};

void print_failure_json(const char* stage, const char* message) {
  printf("{\n");
  printf("  \"schema\": \"mamba3_wave9_runtime_probe_v1\",\n");
  printf("  \"status\": \"fail\",\n");
  printf("  \"kernel_name\": \"mamba3_wave9_uint4_copy_12tile_probe\",\n");
  printf("  \"correctness\": {\"status\": \"not_run\"},\n");
  printf("  \"timing\": {\"status\": \"not_run\"},\n");
  printf("  \"blockers\": [\"%s: %s\"]\n", stage, message);
  printf("}\n");
}

bool check_cuda(cudaError_t error, const char* stage) {
  if (error == cudaSuccess) {
    return true;
  }
  print_failure_json(stage, cudaGetErrorString(error));
  return false;
}

uint16_t pattern_value(int chunk, int tile, int elem, uint32_t salt) {
  uint32_t x = static_cast<uint32_t>(chunk + 1) * 0x9e3779b9u;
  x ^= static_cast<uint32_t>(tile + 17) * 0x85ebca6bu;
  x ^= static_cast<uint32_t>(elem + 29) * 0xc2b2ae35u;
  x ^= salt;
  x ^= x >> 16;
  return static_cast<uint16_t>(x & 0xffffu);
}

uint64_t fnv1a_u16(const std::vector<uint16_t>& values) {
  uint64_t hash = 1469598103934665603ull;
  for (uint16_t value : values) {
    hash ^= static_cast<uint8_t>(value & 0xffu);
    hash *= 1099511628211ull;
    hash ^= static_cast<uint8_t>((value >> 8) & 0xffu);
    hash *= 1099511628211ull;
  }
  return hash;
}

bool parse_int_arg(const char* arg, const char* prefix, int* out) {
  const size_t len = strlen(prefix);
  if (strncmp(arg, prefix, len) != 0) {
    return false;
  }
  *out = atoi(arg + len);
  return true;
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    parse_int_arg(argv[i], "--chunks=", &options.chunks) ||
        parse_int_arg(argv[i], "--warmup=", &options.warmup_iterations) ||
        parse_int_arg(argv[i], "--iters=", &options.timed_iterations) ||
        parse_int_arg(argv[i], "--block-threads=", &options.block_threads) ||
        parse_int_arg(argv[i], "--grid-blocks=", &options.grid_blocks);
  }
  options.chunks = std::max(options.chunks, 1);
  options.warmup_iterations = std::max(options.warmup_iterations, 0);
  options.timed_iterations = std::max(options.timed_iterations, 1);
  options.block_threads = std::max(options.block_threads, 32);
  return options;
}

}  // namespace

extern "C" __global__ void mamba3_wave9_uint4_copy_12tile_probe(
    const __nv_bfloat16* __restrict__ global_src,
    const __nv_bfloat16* __restrict__ local_stage_src,
    __nv_bfloat16* __restrict__ dst,
    int chunks,
    int* __restrict__ status) {
  extern __shared__ __align__(16) unsigned char smem_bytes[];

  auto* smem_vectors = reinterpret_cast<uint4*>(smem_bytes);
  auto* global_vectors = reinterpret_cast<const uint4*>(global_src);
  auto* local_stage_vectors = reinterpret_cast<const uint4*>(local_stage_src);
  auto* dst_vectors = reinterpret_cast<uint4*>(dst);

  __shared__ int block_alignment_failed;
  if (threadIdx.x == 0) {
    const uintptr_t alignment_bits =
        reinterpret_cast<uintptr_t>(global_src) |
        reinterpret_cast<uintptr_t>(local_stage_src) |
        reinterpret_cast<uintptr_t>(dst) |
        reinterpret_cast<uintptr_t>(smem_bytes);
    block_alignment_failed = ((alignment_bits & (kVectorBytes - 1)) != 0);
    if (block_alignment_failed) {
      atomicOr(status, 1);
    }
  }
  __syncthreads();
  if (block_alignment_failed) {
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

    __syncthreads();
  }
}

extern "C" __global__ void mamba3_wave9_scalar_copy_12tile_reference(
    const uint16_t* __restrict__ global_src,
    const uint16_t* __restrict__ local_stage_src,
    uint16_t* __restrict__ dst,
    int chunks) {
  const size_t total_elements = static_cast<size_t>(chunks) * kTotalElements;
  const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

  for (size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total_elements;
       idx += stride) {
    const size_t elem_in_chunk = idx % kTotalElements;
    const int chunk = static_cast<int>(idx / kTotalElements);
    const int tile = static_cast<int>(elem_in_chunk / kTileElements);
    const int elem = static_cast<int>(elem_in_chunk % kTileElements);

    const uint16_t* src = nullptr;
    size_t src_offset = 0;
    if (tile < kGlobalTileCount) {
      src = global_src;
      src_offset =
          (static_cast<size_t>(chunk) * kGlobalTileCount + tile) * kTileElements;
    } else {
      src = local_stage_src;
      src_offset = (static_cast<size_t>(chunk) * kLocalStageTileCount +
                    (tile - kGlobalTileCount)) *
                   kTileElements;
    }
    dst[idx] = src[src_offset + elem];
  }
}

int main(int argc, char** argv) {
  const Options options = parse_options(argc, argv);

  int device = 0;
  cudaDeviceProp props{};
  if (!check_cuda(cudaGetDevice(&device), "cudaGetDevice")) {
    return 2;
  }
  if (!check_cuda(cudaGetDeviceProperties(&props, device), "cudaGetDeviceProperties")) {
    return 2;
  }

  int max_dynamic_smem_optin = 0;
  if (!check_cuda(
          cudaDeviceGetAttribute(
              &max_dynamic_smem_optin,
              cudaDevAttrMaxSharedMemoryPerBlockOptin,
              device),
          "cudaDeviceGetAttribute(max dynamic shared memory opt-in)")) {
    return 2;
  }
  if (max_dynamic_smem_optin < kDynamicSmemBytes) {
    print_failure_json("dynamic_smem_cap", "device cannot opt in to 98304 bytes of dynamic shared memory");
    return 2;
  }

  if (!check_cuda(
          cudaFuncSetAttribute(
              mamba3_wave9_uint4_copy_12tile_probe,
              cudaFuncAttributeMaxDynamicSharedMemorySize,
              kDynamicSmemBytes),
          "cudaFuncSetAttribute(max dynamic shared memory)")) {
    return 2;
  }
  if (!check_cuda(
          cudaFuncSetAttribute(
              mamba3_wave9_uint4_copy_12tile_probe,
              cudaFuncAttributePreferredSharedMemoryCarveout,
              cudaSharedmemCarveoutMaxShared),
          "cudaFuncSetAttribute(shared memory carveout)")) {
    return 2;
  }

  const int grid_blocks =
      options.grid_blocks > 0
          ? options.grid_blocks
          : std::max(1, std::min(options.chunks, props.multiProcessorCount * 2));

  const size_t global_elements =
      static_cast<size_t>(options.chunks) * kGlobalTileCount * kTileElements;
  const size_t local_elements =
      static_cast<size_t>(options.chunks) * kLocalStageTileCount * kTileElements;
  const size_t dst_elements =
      static_cast<size_t>(options.chunks) * kLogicalTileCount * kTileElements;
  const size_t global_bytes = global_elements * sizeof(uint16_t);
  const size_t local_bytes = local_elements * sizeof(uint16_t);
  const size_t dst_bytes = dst_elements * sizeof(uint16_t);

  std::vector<uint16_t> h_global(global_elements);
  std::vector<uint16_t> h_local(local_elements);
  std::vector<uint16_t> h_vector(dst_elements);
  std::vector<uint16_t> h_scalar(dst_elements);

  for (int chunk = 0; chunk < options.chunks; ++chunk) {
    for (int tile = 0; tile < kGlobalTileCount; ++tile) {
      const size_t base =
          (static_cast<size_t>(chunk) * kGlobalTileCount + tile) * kTileElements;
      for (int elem = 0; elem < kTileElements; ++elem) {
        h_global[base + elem] = pattern_value(chunk, tile, elem, 0x13579bdfu);
      }
    }
    for (int tile = 0; tile < kLocalStageTileCount; ++tile) {
      const size_t base =
          (static_cast<size_t>(chunk) * kLocalStageTileCount + tile) *
          kTileElements;
      for (int elem = 0; elem < kTileElements; ++elem) {
        h_local[base + elem] =
            pattern_value(chunk, tile + kGlobalTileCount, elem, 0x2468ace0u);
      }
    }
  }

  uint16_t* d_global = nullptr;
  uint16_t* d_local = nullptr;
  uint16_t* d_vector = nullptr;
  uint16_t* d_scalar = nullptr;
  int* d_status = nullptr;

  if (!check_cuda(cudaMalloc(&d_global, global_bytes), "cudaMalloc(global_src)") ||
      !check_cuda(cudaMalloc(&d_local, local_bytes), "cudaMalloc(local_stage_src)") ||
      !check_cuda(cudaMalloc(&d_vector, dst_bytes), "cudaMalloc(vector_dst)") ||
      !check_cuda(cudaMalloc(&d_scalar, dst_bytes), "cudaMalloc(scalar_dst)") ||
      !check_cuda(cudaMalloc(&d_status, sizeof(int)), "cudaMalloc(status)")) {
    return 2;
  }

  if (!check_cuda(cudaMemcpy(d_global, h_global.data(), global_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(global_src)") ||
      !check_cuda(cudaMemcpy(d_local, h_local.data(), local_bytes, cudaMemcpyHostToDevice), "cudaMemcpy(local_stage_src)") ||
      !check_cuda(cudaMemset(d_vector, 0, dst_bytes), "cudaMemset(vector_dst)") ||
      !check_cuda(cudaMemset(d_scalar, 0, dst_bytes), "cudaMemset(scalar_dst)") ||
      !check_cuda(cudaMemset(d_status, 0, sizeof(int)), "cudaMemset(status)")) {
    return 2;
  }

  dim3 grid(grid_blocks);
  dim3 block(options.block_threads);

  mamba3_wave9_scalar_copy_12tile_reference<<<grid, block>>>(
      d_global, d_local, d_scalar, options.chunks);
  if (!check_cuda(cudaGetLastError(), "launch scalar reference")) {
    return 2;
  }

  mamba3_wave9_uint4_copy_12tile_probe<<<
      grid, block, kDynamicSmemBytes>>>(
      reinterpret_cast<const __nv_bfloat16*>(d_global),
      reinterpret_cast<const __nv_bfloat16*>(d_local),
      reinterpret_cast<__nv_bfloat16*>(d_vector),
      options.chunks,
      d_status);
  if (!check_cuda(cudaGetLastError(), "launch uint4 vector probe") ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(correctness)")) {
    return 2;
  }

  int status_word = 0;
  if (!check_cuda(cudaMemcpy(&status_word, d_status, sizeof(int), cudaMemcpyDeviceToHost), "cudaMemcpy(status)") ||
      !check_cuda(cudaMemcpy(h_vector.data(), d_vector, dst_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(vector_dst)") ||
      !check_cuda(cudaMemcpy(h_scalar.data(), d_scalar, dst_bytes, cudaMemcpyDeviceToHost), "cudaMemcpy(scalar_dst)")) {
    return 2;
  }

  size_t mismatch_count = 0;
  size_t first_mismatch = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < dst_elements; ++i) {
    if (h_vector[i] != h_scalar[i]) {
      if (first_mismatch == std::numeric_limits<size_t>::max()) {
        first_mismatch = i;
      }
      ++mismatch_count;
    }
  }

  for (int i = 0; i < options.warmup_iterations; ++i) {
    mamba3_wave9_uint4_copy_12tile_probe<<<
        grid, block, kDynamicSmemBytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_global),
        reinterpret_cast<const __nv_bfloat16*>(d_local),
        reinterpret_cast<__nv_bfloat16*>(d_vector),
        options.chunks,
        d_status);
  }
  if (!check_cuda(cudaGetLastError(), "launch vector warmup") ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(vector warmup)")) {
    return 2;
  }

  cudaEvent_t vector_start = nullptr;
  cudaEvent_t vector_stop = nullptr;
  cudaEvent_t scalar_start = nullptr;
  cudaEvent_t scalar_stop = nullptr;
  if (!check_cuda(cudaEventCreate(&vector_start), "cudaEventCreate(vector_start)") ||
      !check_cuda(cudaEventCreate(&vector_stop), "cudaEventCreate(vector_stop)") ||
      !check_cuda(cudaEventCreate(&scalar_start), "cudaEventCreate(scalar_start)") ||
      !check_cuda(cudaEventCreate(&scalar_stop), "cudaEventCreate(scalar_stop)")) {
    return 2;
  }

  if (!check_cuda(cudaEventRecord(vector_start), "cudaEventRecord(vector_start)")) {
    return 2;
  }
  for (int i = 0; i < options.timed_iterations; ++i) {
    mamba3_wave9_uint4_copy_12tile_probe<<<
        grid, block, kDynamicSmemBytes>>>(
        reinterpret_cast<const __nv_bfloat16*>(d_global),
        reinterpret_cast<const __nv_bfloat16*>(d_local),
        reinterpret_cast<__nv_bfloat16*>(d_vector),
        options.chunks,
        d_status);
  }
  if (!check_cuda(cudaGetLastError(), "launch vector timing") ||
      !check_cuda(cudaEventRecord(vector_stop), "cudaEventRecord(vector_stop)") ||
      !check_cuda(cudaEventSynchronize(vector_stop), "cudaEventSynchronize(vector_stop)")) {
    return 2;
  }

  for (int i = 0; i < options.warmup_iterations; ++i) {
    mamba3_wave9_scalar_copy_12tile_reference<<<grid, block>>>(
        d_global, d_local, d_scalar, options.chunks);
  }
  if (!check_cuda(cudaGetLastError(), "launch scalar warmup") ||
      !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize(scalar warmup)")) {
    return 2;
  }

  if (!check_cuda(cudaEventRecord(scalar_start), "cudaEventRecord(scalar_start)")) {
    return 2;
  }
  for (int i = 0; i < options.timed_iterations; ++i) {
    mamba3_wave9_scalar_copy_12tile_reference<<<grid, block>>>(
        d_global, d_local, d_scalar, options.chunks);
  }
  if (!check_cuda(cudaGetLastError(), "launch scalar timing") ||
      !check_cuda(cudaEventRecord(scalar_stop), "cudaEventRecord(scalar_stop)") ||
      !check_cuda(cudaEventSynchronize(scalar_stop), "cudaEventSynchronize(scalar_stop)")) {
    return 2;
  }

  float vector_elapsed_ms = 0.0f;
  float scalar_elapsed_ms = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&vector_elapsed_ms, vector_start, vector_stop), "cudaEventElapsedTime(vector)") ||
      !check_cuda(cudaEventElapsedTime(&scalar_elapsed_ms, scalar_start, scalar_stop), "cudaEventElapsedTime(scalar)")) {
    return 2;
  }

  const double vector_avg_us =
      static_cast<double>(vector_elapsed_ms) * 1000.0 / options.timed_iterations;
  const double scalar_avg_us =
      static_cast<double>(scalar_elapsed_ms) * 1000.0 / options.timed_iterations;
  const double logical_payload_bytes_per_iteration = static_cast<double>(dst_bytes);
  const double copy_stage_bytes_per_iteration =
      static_cast<double>(dst_bytes) * 2.0;
  const double gib = 1024.0 * 1024.0 * 1024.0;
  const double vector_gib_s =
      (copy_stage_bytes_per_iteration / gib) / (vector_avg_us * 1.0e-6);
  const double scalar_gib_s =
      (logical_payload_bytes_per_iteration / gib) / (scalar_avg_us * 1.0e-6);
  const uint64_t vector_checksum = fnv1a_u16(h_vector);
  const uint64_t scalar_checksum = fnv1a_u16(h_scalar);
  const bool correctness_pass =
      status_word == 0 && mismatch_count == 0 && vector_checksum == scalar_checksum;

  printf("{\n");
  printf("  \"schema\": \"mamba3_wave9_runtime_probe_v1\",\n");
  printf("  \"status\": \"%s\",\n", correctness_pass ? "pass" : "fail");
  printf("  \"kernel_name\": \"mamba3_wave9_uint4_copy_12tile_probe\",\n");
  printf("  \"scalar_reference_kernel\": \"mamba3_wave9_scalar_copy_12tile_reference\",\n");
  printf("  \"device\": {\n");
  printf("    \"name\": \"%s\",\n", props.name);
  printf("    \"compute_capability\": \"%d.%d\",\n", props.major, props.minor);
  printf("    \"multiprocessor_count\": %d,\n", props.multiProcessorCount);
  printf("    \"max_dynamic_smem_optin_bytes\": %d\n", max_dynamic_smem_optin);
  printf("  },\n");
  printf("  \"constants\": {\n");
  printf("    \"logical_tile_count\": %d,\n", kLogicalTileCount);
  printf("    \"global_tile_count\": %d,\n", kGlobalTileCount);
  printf("    \"local_stage_tile_count\": %d,\n", kLocalStageTileCount);
  printf("    \"tile_rows\": %d,\n", kTileRows);
  printf("    \"tile_cols\": %d,\n", kTileCols);
  printf("    \"dtype\": \"bf16\",\n");
  printf("    \"dtype_bytes\": %d,\n", kBf16Bytes);
  printf("    \"vector_type\": \"uint4\",\n");
  printf("    \"vector_bytes\": %d,\n", kVectorBytes);
  printf("    \"vectors_per_tile\": %d,\n", kVectorsPerTile);
  printf("    \"vectors_per_chunk\": %d,\n", kTotalVectors);
  printf("    \"copy_bytes_per_chunk\": %d,\n", kDynamicSmemBytes);
  printf("    \"dynamic_smem_bytes\": %d\n", kDynamicSmemBytes);
  printf("  },\n");
  printf("  \"launch\": {\n");
  printf("    \"chunks\": %d,\n", options.chunks);
  printf("    \"grid_blocks\": %d,\n", grid_blocks);
  printf("    \"block_threads\": %d,\n", options.block_threads);
  printf("    \"dynamic_smem_bytes\": %d\n", kDynamicSmemBytes);
  printf("  },\n");
  printf("  \"correctness\": {\n");
  printf("    \"status\": \"%s\",\n", correctness_pass ? "pass" : "fail");
  printf("    \"comparison\": \"byte_equal_to_scalar_cuda_kernel\",\n");
  printf("    \"status_word\": %d,\n", status_word);
  printf("    \"mismatched_elements\": %zu,\n", mismatch_count);
  if (first_mismatch == std::numeric_limits<size_t>::max()) {
    printf("    \"first_mismatch_index\": null,\n");
  } else {
    printf("    \"first_mismatch_index\": %zu,\n", first_mismatch);
  }
  printf("    \"vector_checksum_fnv1a64\": %llu,\n", static_cast<unsigned long long>(vector_checksum));
  printf("    \"scalar_checksum_fnv1a64\": %llu\n", static_cast<unsigned long long>(scalar_checksum));
  printf("  },\n");
  printf("  \"timing\": {\n");
  printf("    \"status\": \"measured\",\n");
  printf("    \"warmup_iterations\": %d,\n", options.warmup_iterations);
  printf("    \"timed_iterations\": %d,\n", options.timed_iterations);
  printf("    \"logical_payload_bytes_per_iteration\": %.0f,\n", logical_payload_bytes_per_iteration);
  printf("    \"copy_stage_bytes_per_iteration\": %.0f,\n", copy_stage_bytes_per_iteration);
  printf("    \"vector_avg_us\": %.6f,\n", vector_avg_us);
  printf("    \"scalar_avg_us\": %.6f,\n", scalar_avg_us);
  printf("    \"speedup_vs_scalar_time\": %.6f,\n", scalar_avg_us / vector_avg_us);
  printf("    \"vector_effective_gib_s_copy_stage_bytes\": %.6f,\n", vector_gib_s);
  printf("    \"scalar_effective_gib_s_payload_bytes\": %.6f\n", scalar_gib_s);
  printf("  },\n");
  if (correctness_pass) {
    printf("  \"blockers\": []\n");
  } else {
    printf("  \"blockers\": [\"vector output does not match scalar reference or alignment guard failed\"]\n");
  }
  printf("}\n");

  cudaEventDestroy(vector_start);
  cudaEventDestroy(vector_stop);
  cudaEventDestroy(scalar_start);
  cudaEventDestroy(scalar_stop);
  cudaFree(d_global);
  cudaFree(d_local);
  cudaFree(d_vector);
  cudaFree(d_scalar);
  cudaFree(d_status);

  return correctness_pass ? 0 : 1;
}
