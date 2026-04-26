#include <torch/extension.h>

at::Tensor cutlass_mxfp8_tn_gemm_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "tn_gemm",
      &cutlass_mxfp8_tn_gemm_cuda,
      "CUTLASS SM120/SM121 MXFP8 TN GEMM with native scale prepack");
}
