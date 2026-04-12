"""TileLang sparse MLA kernels from tile-ai/tilelang/examples/deepseek_v32/.

Provides fused sparse attention forward+backward that gathers only topk
K/V entries per query into shared memory, then computes Q@K^T + online
softmax + S@V without materializing the full [b, h, sq, sk] scores tensor.

Source: https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32
License: Apache-2.0 (same as tilelang)
"""
