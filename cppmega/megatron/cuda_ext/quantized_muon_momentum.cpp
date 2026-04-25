#include <torch/extension.h>

#include <vector>

void qmuon_update_multi_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    double beta,
    bool unsigned_storage);

at::Tensor qmuon_update_multi_with_sumsq_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    double beta,
    bool unsigned_storage);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "update_multi_",
      &qmuon_update_multi_cuda_,
      "Blockwise quantized Muon momentum update, writing updated momentum into grad");
  m.def(
      "update_multi_with_sumsq_",
      &qmuon_update_multi_with_sumsq_cuda_,
      "Blockwise quantized Muon momentum update plus per-block sumsq for NS normalization");
}
