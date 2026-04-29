#include <torch/extension.h>

at::Tensor cutlass_mxfp8_tn_gemm_compact_scale_cuda(
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

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_asym_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_asym_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_b_tma_early_asym_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    int64_t a_source,
    int64_t a_data_ld,
    int64_t a_scale_ld,
    int64_t b_source,
    int64_t b_data_ld,
    int64_t b_scale_ld,
    at::Tensor out,
    bool use_out,
    bool accumulate,
    double alpha,
    double beta);

at::Tensor cutlass_mxfp8_tn_gemm_swizzled_scale_cuda(
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

at::Tensor cutlass_mxfp8_tn_gemm_swizzled_scale_strided_cuda(
    at::Tensor A_u8,
    at::Tensor SFA_u8,
    at::Tensor B_u8,
    at::Tensor SFB_u8,
    int64_t m,
    int64_t n,
    int64_t k,
    at::Tensor out,
    int64_t out_ld,
    int64_t out_offset,
    bool accumulate,
    double alpha,
    double beta);

void cutlass_mxfp8_prepare_wgrad_stock_a_tile_cuda(
    at::Tensor dy_colwise_u8,
    at::Tensor dy_colwise_scale_u8,
    at::Tensor a_tile_u8,
    at::Tensor sfa_tile_u8,
    int64_t m_start,
    int64_t tile_m,
    int64_t k,
    int64_t dy_data_ld,
    int64_t dy_scale_ld);

void cutlass_mxfp8_prepare_wgrad_stock_b_scale_tile_cuda(
    at::Tensor x_t_rowwise_scale_u8,
    at::Tensor sfb_tile_u8,
    int64_t n_start,
    int64_t tile_n,
    int64_t k,
    int64_t x_t_scale_ld);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "tn_gemm_compact_scale",
      &cutlass_mxfp8_tn_gemm_compact_scale_cuda,
      "CUTLASS SM120/SM121 MXFP8 TN GEMM with compact scale mainloop loads");
  m.def(
      "tn_gemm_compact_direct",
      &cutlass_mxfp8_tn_gemm_compact_direct_cuda,
      "CUTLASS SM120/SM121 MXFP8 TN GEMM with manual compact payload and scale loads");
  m.def(
      "tn_gemm_compact_direct_asym",
      &cutlass_mxfp8_tn_gemm_compact_direct_asym_cuda,
      "Experimental CUTLASS SM120/SM121 MXFP8 TN GEMM with split MK/NK compact direct loads");
  m.def(
      "tn_gemm_compact_direct_a_col_smem_asym",
      &cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_asym_cuda,
      "Experimental CUTLASS SM120/SM121 MXFP8 TN GEMM with A-columnwise compact loads into an M-contiguous A shared-memory layout");
  m.def(
      "tn_gemm_compact_direct_a_col_smem_b_tma_early_asym",
      &cutlass_mxfp8_tn_gemm_compact_direct_a_col_smem_b_tma_early_asym_cuda,
      "Experimental CUTLASS SM120/SM121 MXFP8 TN GEMM with A-columnwise compact shared-memory layout and early rowwise-B TMA issue");
  m.def(
      "tn_gemm_swizzled_scale",
      &cutlass_mxfp8_tn_gemm_swizzled_scale_cuda,
      "Probe-only CUTLASS SM120/SM121 MXFP8 TN GEMM with rowwise payloads and GEMM-swizzled scale layouts");
  m.def(
      "tn_gemm_swizzled_scale_strided",
      &cutlass_mxfp8_tn_gemm_swizzled_scale_strided_cuda,
      "Probe-only CUTLASS SM120/SM121 MXFP8 TN GEMM with strided row-major output");
  m.def(
      "prepare_wgrad_stock_a_tile",
      &cutlass_mxfp8_prepare_wgrad_stock_a_tile_cuda,
      "Prepare one dy.T tile payload and GEMM-swizzled scale scratch for the stock MXFP8 GEMM");
  m.def(
      "prepare_wgrad_stock_b_scale_tile",
      &cutlass_mxfp8_prepare_wgrad_stock_b_scale_tile_cuda,
      "Prepare one x.T GEMM-swizzled scale scratch tile for the stock MXFP8 GEMM");
}
