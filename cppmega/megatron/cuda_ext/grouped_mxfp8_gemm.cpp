#include <torch/extension.h>

at::Tensor grouped_mxfp8_dgrad_nn_cuda(
    at::Tensor dy_u8,
    at::Tensor sf_dy_u8,
    at::Tensor weight_u8,
    at::Tensor sf_weight_u8,
    at::Tensor expert_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor grouped_mxfp8_wgrad_nt_cuda(
    at::Tensor dy_u8,
    at::Tensor sf_dy_u8,
    at::Tensor x_u8,
    at::Tensor sf_x_u8,
    at::Tensor expert_offsets,
    at::Tensor scale_offsets,
    bool use_scale_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor grouped_mxfp8_dgrad_nn_ptrs_cuda(
    at::Tensor dy_ptrs,
    at::Tensor sf_dy_ptrs,
    at::Tensor weight_ptrs,
    at::Tensor sf_weight_ptrs,
    at::Tensor expert_offsets,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta,
    int64_t total_rows,
    int64_t n,
    int64_t k);

void grouped_mxfp8_wgrad_nt_ptrs_cuda(
    at::Tensor dy_ptrs,
    at::Tensor sf_dy_ptrs,
    at::Tensor x_ptrs,
    at::Tensor sf_x_ptrs,
    at::Tensor out_ptrs,
    at::Tensor expert_offsets,
    bool accumulate,
    double alpha,
    double beta,
    int64_t n,
    int64_t k);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "dgrad_nn",
      &grouped_mxfp8_dgrad_nn_cuda,
      "Grouped MXFP8 dgrad NN direct compact reference kernel");
  m.def(
      "wgrad_nt",
      &grouped_mxfp8_wgrad_nt_cuda,
      "Grouped MXFP8 wgrad NT direct compact reference kernel");
  m.def(
      "dgrad_nn_ptrs",
      &grouped_mxfp8_dgrad_nn_ptrs_cuda,
      "Grouped MXFP8 dgrad NN direct compact per-expert pointer kernel");
  m.def(
      "wgrad_nt_ptrs",
      &grouped_mxfp8_wgrad_nt_ptrs_cuda,
      "Grouped MXFP8 wgrad NT direct compact per-expert pointer kernel");
}
