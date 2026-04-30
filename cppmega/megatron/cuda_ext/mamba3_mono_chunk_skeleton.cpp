#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> mamba3_mono_chunk_skeleton_cuda(
    at::Tensor q,
    at::Tensor k,
    at::Tensor dout,
    at::Tensor v,
    at::Tensor mimo_v,
    at::Tensor mimo_o,
    at::Tensor qk_dot,
    at::Tensor dt,
    at::Tensor trap,
    at::Tensor dstates,
    int64_t chunk_size);

py::dict mamba3_mono_chunk_skeleton_metadata();

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "mono_chunk_skeleton",
      &mamba3_mono_chunk_skeleton_cuda,
      "Mamba3 bwd_bwd monolithic chunk CUDA skeleton");
  m.def(
      "kernel_metadata",
      &mamba3_mono_chunk_skeleton_metadata,
      "Static metadata for the Mamba3 monolithic chunk CUDA skeleton");
}
