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

at::Tensor qmuon_update_multi_with_group_sumsq_cuda_(
    std::vector<at::Tensor> q_tensors,
    std::vector<at::Tensor> absmax_tensors,
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    int64_t num_groups,
    double beta,
    bool unsigned_storage);

at::Tensor qmuon_update_with_sumsq_no_grad_cuda_(
    at::Tensor q,
    at::Tensor absmax,
    at::Tensor grad,
    double beta,
    bool unsigned_storage);

void qmuon_emit_mxfp8_carrier_from_quantized_cuda_(
    at::Tensor q,
    at::Tensor absmax,
    at::Tensor rowwise_data,
    at::Tensor rowwise_scale,
    at::Tensor inv_norm,
    bool unsigned_storage);

void qmuon_scale_multi_by_group_cuda_(
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    at::Tensor inv_norms);

void qmuon_scale_multi_by_group_from_sumsq_cuda_(
    std::vector<at::Tensor> grad_tensors,
    at::Tensor block_group_ids,
    at::Tensor group_sumsq,
    double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "update_multi_",
      &qmuon_update_multi_cuda_,
      "Blockwise quantized Muon momentum update, writing updated momentum into grad");
  m.def(
      "update_multi_with_sumsq_",
      &qmuon_update_multi_with_sumsq_cuda_,
      "Blockwise quantized Muon momentum update plus per-block sumsq for NS normalization");
  m.def(
      "update_multi_with_group_sumsq_",
      &qmuon_update_multi_with_group_sumsq_cuda_,
      "Blockwise quantized Muon momentum update plus grouped sumsq for sliced NS normalization");
  m.def(
      "update_with_sumsq_no_grad_",
      &qmuon_update_with_sumsq_no_grad_cuda_,
      "Blockwise quantized Muon momentum update plus per-block sumsq without writing grad scratch");
  m.def(
      "emit_mxfp8_carrier_from_quantized_",
      &qmuon_emit_mxfp8_carrier_from_quantized_cuda_,
      "Emit rowwise MXFP8 carrier payload/scales from quantized Muon momentum");
  m.def(
      "scale_multi_by_group_",
      &qmuon_scale_multi_by_group_cuda_,
      "Scale updated momentum grads in place by per-group inverse norms");
  m.def(
      "scale_multi_by_group_from_sumsq_",
      &qmuon_scale_multi_by_group_from_sumsq_cuda_,
      "Scale updated momentum grads in place by per-group sumsq values");
}
