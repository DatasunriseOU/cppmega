"""Modal app: DSA indexer kernel-level micro-bench on B200 (Stream F, task #82).

Purpose
-------
Standalone kernel-level micro-benchmark of the DeepSeek-V3.2-Exp DSA
(Lightning Indexer) compute on Modal B200 (sm_100, full Blackwell).
Compare FP4 vs FP8 vs BF16 latency + memory for the indexer inner loop
(4 replicated linear sub-modules + index-score einsum + topk).

This is **NOT** a training run. It does not touch megatron-core, does not
pull Stream E's FP8 port, and does not mutate any source. It inlines the
DSA indexer math exactly as written in
`megatron/core/transformer/experimental_attention_variant/dsa.py`
(bench3 ref, fetched 2026-04-12, 1119 LOC) — the 4 linears, LayerNorm,
`_compute_index_scores`, and `fused_qk_topk_naive` — then times each
variant across BF16 / FP8 / FP4 (if available on sm_100 TE 2.13).

Shape (hard-coded to the NAM56R production config):
    hidden_size     = 3584
    q_lora_rank     = 64
    index_n_heads   = 8
    index_head_dim  = 64
    batch           = 4
    seqlen          = 4096

Usage
-----
    modal run scripts/modal_dsa_indexer_bench.py::bench
    modal run scripts/modal_dsa_indexer_bench.py       # same, via main

Cost: single B200 on Modal is ~$10-15/hr. The `bench` function caps at
600 s wall-clock via timeout=600. Do NOT leave this running.
"""
# ruff: noqa: E402

from __future__ import annotations

import os
from typing import Any

import modal

_PYTHON = "3.13"
_TORCH_NIGHTLY_INDEX = "https://download.pytorch.org/whl/nightly/cu132"

# B200 single-GPU — DSA indexer is replicated, not TP-split.
_GPU_SPEC = os.environ.get("CPPMEGA_MODAL_GPU", "B200:1")

app = modal.App("cppmega-dsa-indexer-bench")


def _image() -> modal.Image:
    """Build the DSA indexer bench image for B200 sm_100.

    Uses NVIDIA NGC PyTorch container as the base — this ships torch,
    cuDNN, cuBLAS, and Transformer Engine already linked and tested
    together. Avoids the source-build path that fails on cudnn.h missing
    from plain `nvidia-cuda-nvcc` installs.

    nvcr.io/nvidia/pytorch:25.03-py3 ships:
      - torch 2.7.x (NGC's fork, stable)
      - cuDNN 9.x
      - cuBLAS 12.8
      - Transformer Engine 2.0+ with FP8 HYBRID recipe and (on 25.03+)
        preliminary FP4 NVFP4/MXFP4 support for Blackwell sm_100
      - Python 3.12

    We do NOT install megatron-core — DSA indexer math is inlined.
    We do NOT install fast_hadamard_transform — DSA uses it for a BF16
    rotation that we replace with a pure-torch FWHT (same math for
    power-of-2 last dim; applies to q and k identically in all variants
    so it cancels out of the delta).
    """
    # NGC PyTorch container tags: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch/tags
    # 25.03 targets Blackwell-ready stack with TE 2.x. Modal needs the
    # `add_python` escape because NGC ships its own Python in /usr/bin.
    base: Any = modal.Image.from_registry(
        "nvcr.io/nvidia/pytorch:25.03-py3",
        add_python=None,  # use the container's existing Python
    )
    img = (
        base.run_commands(
            "which python && python --version",
            "python -c 'import torch; print(\"torch\", torch.__version__, torch.version.cuda)'",
            "python -c 'import transformer_engine as te; print(\"te\", getattr(te, \"__version__\", \"unknown\"))' "
            "|| echo 'TE not preinstalled — will need runtime install'",
            "python -c 'import transformer_engine.pytorch as tep; print(\"tep ok\")' "
            "|| echo 'TE pytorch bindings not loadable'",
        )
    )
    return img


image = _image()


# ---------------------------------------------------------------------------
# Inlined DSA indexer math (copied from megatron dsa.py, bench3, 2026-04-12)
# ---------------------------------------------------------------------------
#
# We copy just the helpers we need: `_compute_index_scores` and
# `fused_qk_topk_naive`. These must be importable on the Modal side; we
# define them inside the function body so there's no cross-container
# serialization concern.


_DSA_HELPERS_SRC = '''
import torch
import torch.nn.functional as F


def rotate_activation_bf16(x: torch.Tensor) -> torch.Tensor:
    """BF16 Hadamard rotation — pure-torch FWHT stand-in.

    The real DSA uses fast_hadamard_transform which has no prebuilt wheel
    for our torch 2.12/cu132 image. This is mathematically equivalent for
    power-of-two last-dim sizes (DSA uses index_head_dim=64, a power of 2),
    is applied in BF16 in ALL variants (the dsa.py reference asserts bf16),
    and therefore cancels out of the precision comparison.

    It's slower than the fused kernel but only appears ONCE per forward,
    between the linears and the einsum — we're not benchmarking it, so its
    latency ends up additively inside the `linear_fwd` timer for all
    variants equally.
    """
    assert x.dtype == torch.bfloat16, f"got {x.dtype}"
    n = x.size(-1)
    # n must be a power of 2 — assert it (DSA uses 64).
    assert (n & (n - 1)) == 0 and n > 0, f"FWHT needs power-of-2 last dim, got {n}"
    orig_shape = x.shape
    y = x.reshape(-1, n).float()  # FP32 for stability of the butterflies
    h = 1
    while h < n:
        # Butterfly: y[:, i] += y[:, i+h], y[:, i+h] = y[:, i] - 2*y[:, i+h]
        y_view = y.view(-1, n // (2 * h), 2, h)
        a = y_view[:, :, 0, :].clone()
        b = y_view[:, :, 1, :].clone()
        y_view[:, :, 0, :] = a + b
        y_view[:, :, 1, :] = a - b
        y = y_view.reshape(-1, n)
        h *= 2
    y = y * (n ** -0.5)
    return y.reshape(orig_shape).to(torch.bfloat16)


def compute_index_scores_bf16(q, weights, k):
    """BF16 path: dsa.py::_compute_index_scores exactly (FP32 einsum).

    q:       [sq, b, h, d] bf16
    weights: [sq, b, h]    bf16
    k:       [sk, b, d]    bf16
    returns: [b, sq, sk]   fp32
    """
    # Matches dsa.py line 278 exactly.
    index_scores = torch.einsum("sbhd,tbd->sbht", q.float(), k.float())
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * weights.float().unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)
    return index_scores


def compute_index_scores_lowprec(q, weights, k, compute_dtype):
    """Low-precision path: cast inputs to compute_dtype, run einsum in it,
    accumulate to FP32. For FP8/FP4 we can't do einsum directly (torch
    has no fp8 einsum), so we do it via matmul-in-compute_dtype with the
    accumulator promoted to FP32.

    q, weights, k are BF16 on input — they are the OUTPUTS of TE.Linear
    which, when run inside an fp8_autocast context, internally run the
    GEMM in fp8 but return bf16 activations. This function adds the
    **index-compute** step precision knob.
    """
    # For fp8 / fp4, torch has no native einsum; emulate via cast-and-matmul.
    # This is only the einsum+relu+weighted sum, NOT the linears.
    # We still accumulate in fp32 because the DSA reference does.
    q_c = q.to(compute_dtype).float()
    k_c = k.to(compute_dtype).float()
    w_c = weights.to(compute_dtype).float()
    index_scores = torch.einsum("sbhd,tbd->sbht", q_c, k_c)
    index_scores = torch.relu(index_scores)
    index_scores = index_scores * w_c.unsqueeze(-1)
    index_scores = index_scores.sum(dim=2)
    index_scores = index_scores.transpose(0, 1)
    return index_scores


def fused_qk_topk_naive_custom(q, k, weights, index_topk, compute_fn):
    """Variant of dsa.py::fused_qk_topk_naive with pluggable compute_fn."""
    seqlen = q.size(0)
    index_scores = compute_fn(q, weights, k)
    topk_k = min(index_topk, seqlen)
    topk_indices = index_scores.topk(topk_k, dim=-1)[1]
    return index_scores, topk_indices
'''


@app.function(
    image=image,
    gpu=_GPU_SPEC,
    timeout=600,  # 10 min hard cap
)
def bench() -> dict[str, Any]:
    """Run the DSA indexer micro-bench across BF16 / FP8 / FP4 variants.

    Returns a dict with per-variant timings, peak memory, and topk overlap
    vs the BF16 ground truth.
    """
    import sys
    import time
    import statistics
    from typing import Callable

    import torch
    import torch.nn as nn

    # Execute the DSA helpers module in local scope so compute_fn callables
    # are available with the correct closure.
    _g: dict[str, Any] = {}
    exec(_DSA_HELPERS_SRC, _g)
    compute_index_scores_bf16 = _g["compute_index_scores_bf16"]
    compute_index_scores_lowprec = _g["compute_index_scores_lowprec"]
    fused_qk_topk_naive_custom = _g["fused_qk_topk_naive_custom"]
    rotate_activation_bf16 = _g["rotate_activation_bf16"]

    device = torch.device("cuda:0")
    results: dict[str, Any] = {}

    # ---- stack ----
    results["torch_version"] = torch.__version__
    results["torch_cuda"] = torch.version.cuda
    results["cuda_device"] = torch.cuda.get_device_name(0)
    results["cuda_cap"] = torch.cuda.get_device_capability(0)
    print(f"[bench] torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"[bench] device={results['cuda_device']} cap={results['cuda_cap']}")

    # ---- Transformer Engine import ----
    te_available = False
    te_fp8_available = False
    te_fp4_available = False
    te_version = None
    te_fp8_recipe_cls: Any = None
    te_fp4_recipe_cls: Any = None
    te_fp8_recipe_inst: Any = None
    te_fp4_recipe_inst: Any = None
    te_fp4_recipe_name = None
    try:
        import transformer_engine as te  # noqa: F401
        import transformer_engine.pytorch as tep  # noqa: F401
        from transformer_engine.common import recipe as te_recipe  # type: ignore

        te_available = True
        te_version = getattr(te, "__version__", "unknown")
        print(f"[bench] transformer_engine imported, version={te_version}")

        # FP8 — DelayedScaling is the long-standing API.
        try:
            te_fp8_recipe_cls = te_recipe.DelayedScaling
            te_fp8_recipe_inst = te_fp8_recipe_cls(
                fp8_format=te_recipe.Format.HYBRID,
                amax_history_len=16,
                amax_compute_algo="max",
            )
            te_fp8_available = True
            print("[bench] FP8 recipe: DelayedScaling(HYBRID) available")
        except Exception as exc:  # noqa: BLE001
            print(f"[bench] FP8 recipe NOT available: {type(exc).__name__}: {exc}")

        # FP4 — probe both nvfp4 and mxfp4 recipe class names. TE 2.13 ships
        # these on sm_100 but they may live under different symbol names.
        for cand_attr in ("NVFP4BlockScaling", "MXFP4BlockScaling", "Float4CurrentScaling",
                          "NVFP4", "MXFP4"):
            cls = getattr(te_recipe, cand_attr, None)
            if cls is not None:
                try:
                    inst = cls()
                    te_fp4_recipe_cls = cls
                    te_fp4_recipe_inst = inst
                    te_fp4_recipe_name = cand_attr
                    te_fp4_available = True
                    print(f"[bench] FP4 recipe: {cand_attr} instantiated OK")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[bench] FP4 recipe {cand_attr} present but ctor failed: "
                          f"{type(exc).__name__}: {exc}")
        if not te_fp4_available:
            # Dump all public recipe-module symbols so we know what IS there.
            recipe_syms = [s for s in dir(te_recipe) if not s.startswith("_")]
            print(f"[bench] FP4 recipe: none of the probe names found. "
                  f"te.common.recipe public symbols: {recipe_syms}")

    except Exception as exc:  # noqa: BLE001
        print(f"[bench] transformer_engine import FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)

    results["te_available"] = te_available
    results["te_version"] = te_version
    results["te_fp8_available"] = te_fp8_available
    results["te_fp4_available"] = te_fp4_available
    results["te_fp4_recipe_name"] = te_fp4_recipe_name

    # ---- DSA production shape (NAM56R) ----
    hidden_size = 3584
    q_lora_rank = 64
    index_n_heads = 8
    index_head_dim = 64
    index_topk = 16
    batch = 4
    seqlen = 4096

    results["shape"] = {
        "hidden_size": hidden_size,
        "q_lora_rank": q_lora_rank,
        "index_n_heads": index_n_heads,
        "index_head_dim": index_head_dim,
        "index_topk": index_topk,
        "batch": batch,
        "seqlen": seqlen,
    }
    print(f"[bench] shape: hidden={hidden_size} qlora={q_lora_rank} "
          f"h={index_n_heads} d={index_head_dim} B={batch} S={seqlen}")

    # ---- build the linear sub-modules ----
    # linear_wq_b     : q_lora_rank -> index_n_heads * index_head_dim     (64 -> 512)
    # linear_wk       : hidden_size -> index_head_dim                     (3584 -> 64)
    # k_norm          : LayerNorm(index_head_dim)                         (64)
    # linear_weights_proj : hidden_size -> index_n_heads                  (3584 -> 8)
    #
    # NOTE: linear_weights_proj out=8 is NOT a multiple of 16 — cuBLAS FP8
    # GEMM requires lda % 16 == 0, so TE fp8_autocast will fall back to
    # BF16 for that linear anyway. Same constraint applies to FP4 block
    # scaling (block=16 or 32). We flag this in the results.

    def build_torch_linears(dtype: torch.dtype) -> dict[str, nn.Module]:
        torch.manual_seed(0)
        mods = {
            "linear_wq_b": nn.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False),
            "linear_wk": nn.Linear(hidden_size, index_head_dim, bias=False),
            "k_norm": nn.LayerNorm(index_head_dim, eps=1e-5),
            "linear_weights_proj": nn.Linear(hidden_size, index_n_heads, bias=False),
        }
        for m in mods.values():
            m.to(device=device, dtype=dtype)
        return mods

    def build_te_linears(dtype: torch.dtype) -> dict[str, nn.Module]:
        import transformer_engine.pytorch as tep
        torch.manual_seed(0)
        mods = {
            "linear_wq_b": tep.Linear(q_lora_rank, index_n_heads * index_head_dim, bias=False,
                                      params_dtype=dtype),
            "linear_wk": tep.Linear(hidden_size, index_head_dim, bias=False, params_dtype=dtype),
            # LayerNorm: TE provides tep.LayerNorm which supports BF16/FP32.
            "k_norm": tep.LayerNorm(index_head_dim, eps=1e-5, params_dtype=dtype),
            # linear_weights_proj has out=8 which breaks FP8 lda%16 — keep it torch.
            "linear_weights_proj": nn.Linear(hidden_size, index_n_heads, bias=False),
        }
        mods["linear_weights_proj"].to(device=device, dtype=dtype)
        for k, m in mods.items():
            if k != "linear_weights_proj":
                m.to(device=device)
        return mods

    # ---- fixed random inputs ----
    torch.manual_seed(1)
    x_bf16 = torch.randn(seqlen, batch, hidden_size, device=device, dtype=torch.bfloat16)
    qr_bf16 = torch.randn(seqlen, batch, q_lora_rank, device=device, dtype=torch.bfloat16)

    # ---- one forward+backward pass, generic ----
    def run_forward(
        mods: dict[str, nn.Module],
        x: torch.Tensor,
        qr: torch.Tensor,
        compute_fn: Callable,
        fp8_ctx: Any = None,  # optional context manager
    ):
        def _forward_core():
            # linear_wq_b
            q = mods["linear_wq_b"](qr)
            if isinstance(q, tuple):  # TE Linear returns (out, bias)-style tuple in some versions
                q = q[0]
            # [S, B, H*D] -> [S, B, H, D]
            q = q.reshape(seqlen, batch, index_n_heads, index_head_dim)

            # linear_wk
            k = mods["linear_wk"](x)
            if isinstance(k, tuple):
                k = k[0]
            # k_norm
            k = mods["k_norm"](k)
            if isinstance(k, tuple):
                k = k[0]
            # [S, B, D] -> [S, B, 1, D] -> [S, B, D] (our shape for compute_fn)
            k = k.reshape(seqlen, batch, index_head_dim)

            # Hadamard rotation (BF16 only)
            q = q.to(torch.bfloat16)
            k = k.to(torch.bfloat16)
            q = rotate_activation_bf16(q)
            k = rotate_activation_bf16(k)

            # linear_weights_proj
            w = mods["linear_weights_proj"](x)
            if isinstance(w, tuple):
                w = w[0]
            w = w * (index_n_heads ** -0.5) * (index_head_dim ** -0.5)
            w = w.to(torch.bfloat16)

            index_scores, topk_indices = fused_qk_topk_naive_custom(
                q, k, w, index_topk, compute_fn
            )
            return q, k, w, index_scores, topk_indices

        if fp8_ctx is not None:
            with fp8_ctx:
                return _forward_core()
        return _forward_core()

    # ---- timing helper ----
    def time_op(fn: Callable, warmup: int = 20, iters: int = 100) -> dict[str, float]:
        # Warmup
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        # Per-iter timing via CUDA events
        times_us: list[float] = []
        for _ in range(iters):
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            stop.record()
            torch.cuda.synchronize()
            times_us.append(start.elapsed_time(stop) * 1e3)  # ms -> us
        mean = statistics.fmean(times_us)
        times_sorted = sorted(times_us)
        p50 = times_sorted[len(times_sorted) // 2]
        p99 = times_sorted[int(len(times_sorted) * 0.99)]
        return {"mean_us": mean, "p50_us": p50, "p99_us": p99}

    # ---- variants ----
    variants: list[tuple[str, Callable[[], dict[str, Any]]]] = []

    # BF16 ground truth — all torch, no TE, FP32 index compute (matches dsa.py).
    def run_bf16():
        mods = build_torch_linears(torch.bfloat16)
        x = x_bf16.clone().detach().requires_grad_(True)
        qr = qr_bf16.clone().detach().requires_grad_(True)
        # Fwd-only timing
        def fwd():
            out = run_forward(mods, x, qr, compute_index_scores_bf16, fp8_ctx=None)
            return out
        # Fwd+bwd timing
        def fwd_bwd():
            out = run_forward(mods, x, qr, compute_index_scores_bf16, fp8_ctx=None)
            loss = out[3].float().sum()
            loss.backward()
            for p in mods.values():
                for pp in p.parameters():
                    if pp.grad is not None:
                        pp.grad = None
            x.grad = None
            qr.grad = None
            return out
        # Index-compute-only timing (given precomputed q/k/w)
        q_ref, k_ref, w_ref, _, _ = run_forward(
            mods, x_bf16, qr_bf16, compute_index_scores_bf16, fp8_ctx=None
        )
        def idx_fwd():
            # signature is (q, weights, k) per dsa.py::_compute_index_scores
            return compute_index_scores_bf16(q_ref, w_ref, k_ref)
        def idx_bwd():
            q2 = q_ref.detach().clone().requires_grad_(True)
            k2 = k_ref.detach().clone().requires_grad_(True)
            w2 = w_ref.detach().clone().requires_grad_(True)
            scores = compute_index_scores_bf16(q2, w2, k2)
            scores.sum().backward()
        def topk_only():
            scores = compute_index_scores_bf16(q_ref, w_ref, k_ref)
            return scores.topk(min(index_topk, seqlen), dim=-1)[1]

        torch.cuda.reset_peak_memory_stats()
        t_linear_fwd = time_op(fwd, warmup=10, iters=50)
        t_linear_bwd = time_op(fwd_bwd, warmup=10, iters=50)
        t_idx_fwd = time_op(idx_fwd, warmup=10, iters=100)
        t_idx_bwd = time_op(idx_bwd, warmup=10, iters=50)
        t_topk = time_op(topk_only, warmup=10, iters=100)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # Ground truth topk for later overlap comparison
        _, _, _, _, topk_ref = run_forward(
            mods, x_bf16, qr_bf16, compute_index_scores_bf16, fp8_ctx=None
        )

        return {
            "linear_fwd": t_linear_fwd,
            "linear_bwd": t_linear_bwd,
            "index_compute_fwd": t_idx_fwd,
            "index_compute_bwd": t_idx_bwd,
            "fused_qk_topk": t_topk,
            "peak_memory_MB": peak_mb,
            "topk_ref": topk_ref.detach(),
        }

    variants.append(("BF16", run_bf16))

    # FP8 — TE linears inside fp8_autocast, index compute cast via bf16->fp8->fp32
    def run_fp8():
        if not te_fp8_available:
            return {"status": "skipped_no_te_fp8"}
        import transformer_engine.pytorch as tep
        mods = build_te_linears(torch.bfloat16)  # TE Linear params are bf16, FP8 applies to GEMM inputs
        fp8_ctx_mgr = lambda: tep.fp8_autocast(enabled=True, fp8_recipe=te_fp8_recipe_inst)  # noqa: E731

        # FP8 index compute: torch lacks fp8 einsum; we cast BF16 q,k,w -> fp8 -> fp32
        # and run the FP32 einsum. This measures PRECISION LOSS of FP8 in the
        # index compute, not the kernel speedup — the speedup comes from the
        # linears running FP8 GEMM under TE autocast.
        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            print("[bench] torch.float8_e4m3fn not available; using bf16 for index compute")
            def compute_fn(q, w, k):
                return compute_index_scores_bf16(q, w, k)
        else:
            def compute_fn(q, w, k):
                return compute_index_scores_lowprec(q, w, k, fp8_dtype)

        x = x_bf16.clone().detach().requires_grad_(True)
        qr = qr_bf16.clone().detach().requires_grad_(True)

        def fwd():
            return run_forward(mods, x, qr, compute_fn, fp8_ctx=fp8_ctx_mgr())
        def fwd_bwd():
            out = run_forward(mods, x, qr, compute_fn, fp8_ctx=fp8_ctx_mgr())
            loss = out[3].float().sum()
            loss.backward()
            for m in mods.values():
                for pp in m.parameters():
                    if pp.grad is not None:
                        pp.grad = None
            x.grad = None
            qr.grad = None
            return out
        with fp8_ctx_mgr():
            q_ref, k_ref, w_ref, _, _ = run_forward(mods, x_bf16, qr_bf16, compute_fn, fp8_ctx=None)
        def idx_fwd():
            return compute_fn(q_ref, w_ref, k_ref)
        def idx_bwd():
            q2 = q_ref.detach().clone().requires_grad_(True)
            k2 = k_ref.detach().clone().requires_grad_(True)
            w2 = w_ref.detach().clone().requires_grad_(True)
            scores = compute_fn(q2, w2, k2)
            scores.sum().backward()
        def topk_only():
            scores = compute_fn(q_ref, w_ref, k_ref)
            return scores.topk(min(index_topk, seqlen), dim=-1)[1]

        torch.cuda.reset_peak_memory_stats()
        t_linear_fwd = time_op(fwd, warmup=10, iters=50)
        t_linear_bwd = time_op(fwd_bwd, warmup=10, iters=50)
        t_idx_fwd = time_op(idx_fwd, warmup=10, iters=100)
        t_idx_bwd = time_op(idx_bwd, warmup=10, iters=50)
        t_topk = time_op(topk_only, warmup=10, iters=100)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        # topk for overlap check
        with fp8_ctx_mgr():
            _, _, _, _, topk_fp8 = run_forward(mods, x_bf16, qr_bf16, compute_fn, fp8_ctx=None)

        return {
            "status": "ok",
            "linear_fwd": t_linear_fwd,
            "linear_bwd": t_linear_bwd,
            "index_compute_fwd": t_idx_fwd,
            "index_compute_bwd": t_idx_bwd,
            "fused_qk_topk": t_topk,
            "peak_memory_MB": peak_mb,
            "topk_fp8": topk_fp8.detach(),
        }

    variants.append(("FP8", run_fp8))

    # FP4 — TE linears under fp4_autocast (if available)
    def run_fp4():
        if not te_fp4_available:
            return {"status": "skipped_no_te_fp4"}
        import transformer_engine.pytorch as tep
        mods = build_te_linears(torch.bfloat16)

        # TE exposes fp4 via the same fp8_autocast entry point with an FP4
        # recipe object, or via a dedicated context. Probe for both.
        fp4_ctx_factory = None
        for ctx_name in ("fp4_autocast", "fp8_autocast"):
            fn = getattr(tep, ctx_name, None)
            if fn is not None:
                def _mk(fn=fn):
                    try:
                        return fn(enabled=True, fp8_recipe=te_fp4_recipe_inst)
                    except TypeError:
                        return fn(enabled=True, recipe=te_fp4_recipe_inst)
                try:
                    _test = _mk()
                    _test.__enter__()
                    _test.__exit__(None, None, None)
                    fp4_ctx_factory = _mk
                    print(f"[bench] FP4 context: using tep.{ctx_name} with recipe "
                          f"{te_fp4_recipe_name}")
                    break
                except Exception as exc:  # noqa: BLE001
                    print(f"[bench] FP4 context tep.{ctx_name} failed: "
                          f"{type(exc).__name__}: {exc}")
        if fp4_ctx_factory is None:
            return {"status": "skipped_no_fp4_ctx"}

        fp4_dtype = None
        for cand in ("float4_e2m1fn_x2", "float4_e2m1fn", "uint4"):
            fp4_dtype = getattr(torch, cand, None)
            if fp4_dtype is not None:
                break
        if fp4_dtype is None:
            # Still OK — we can run TE FP4 on the GEMM but emulate the
            # index compute at FP8 (closest available) for the precision
            # comparison. Document in results.
            fp8_dt = getattr(torch, "float8_e4m3fn", None)
            if fp8_dt is None:
                return {"status": "skipped_no_fp4_or_fp8_dtype"}
            def compute_fn(q, w, k):
                return compute_index_scores_lowprec(q, w, k, fp8_dt)
            compute_fn_dtype = "emulated_fp8"
        else:
            def compute_fn(q, w, k):
                return compute_index_scores_lowprec(q, w, k, fp4_dtype)
            compute_fn_dtype = "fp4"

        x = x_bf16.clone().detach().requires_grad_(True)
        qr = qr_bf16.clone().detach().requires_grad_(True)

        def fwd():
            return run_forward(mods, x, qr, compute_fn, fp8_ctx=fp4_ctx_factory())
        def fwd_bwd():
            out = run_forward(mods, x, qr, compute_fn, fp8_ctx=fp4_ctx_factory())
            loss = out[3].float().sum()
            loss.backward()
            for m in mods.values():
                for pp in m.parameters():
                    if pp.grad is not None:
                        pp.grad = None
            x.grad = None
            qr.grad = None
            return out
        with fp4_ctx_factory():
            q_ref, k_ref, w_ref, _, _ = run_forward(mods, x_bf16, qr_bf16, compute_fn, fp8_ctx=None)
        def idx_fwd():
            return compute_fn(q_ref, w_ref, k_ref)
        def idx_bwd():
            q2 = q_ref.detach().clone().requires_grad_(True)
            k2 = k_ref.detach().clone().requires_grad_(True)
            w2 = w_ref.detach().clone().requires_grad_(True)
            scores = compute_fn(q2, w2, k2)
            scores.sum().backward()
        def topk_only():
            scores = compute_fn(q_ref, w_ref, k_ref)
            return scores.topk(min(index_topk, seqlen), dim=-1)[1]

        torch.cuda.reset_peak_memory_stats()
        t_linear_fwd = time_op(fwd, warmup=10, iters=50)
        t_linear_bwd = time_op(fwd_bwd, warmup=10, iters=50)
        t_idx_fwd = time_op(idx_fwd, warmup=10, iters=100)
        t_idx_bwd = time_op(idx_bwd, warmup=10, iters=50)
        t_topk = time_op(topk_only, warmup=10, iters=100)
        peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

        with fp4_ctx_factory():
            _, _, _, _, topk_fp4 = run_forward(mods, x_bf16, qr_bf16, compute_fn, fp8_ctx=None)

        return {
            "status": "ok",
            "compute_fn_dtype": compute_fn_dtype,
            "linear_fwd": t_linear_fwd,
            "linear_bwd": t_linear_bwd,
            "index_compute_fwd": t_idx_fwd,
            "index_compute_bwd": t_idx_bwd,
            "fused_qk_topk": t_topk,
            "peak_memory_MB": peak_mb,
            "topk_fp4": topk_fp4.detach(),
        }

    variants.append(("FP4", run_fp4))

    # ---- run all variants ----
    per_variant: dict[str, Any] = {}
    t_budget_start = time.time()
    for name, fn in variants:
        if time.time() - t_budget_start > 540:  # 9-min soft cap, leave 1 min for cleanup
            print(f"[bench] time budget exceeded, skipping {name}")
            per_variant[name] = {"status": "skipped_time_budget"}
            continue
        print(f"[bench] running variant {name} ...")
        try:
            per_variant[name] = fn()
            print(f"[bench] {name} done")
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            per_variant[name] = {"status": f"error: {type(exc).__name__}: {exc}"}
        torch.cuda.empty_cache()

    # ---- topk overlap ----
    topk_ref = per_variant.get("BF16", {}).get("topk_ref") if isinstance(
        per_variant.get("BF16"), dict) else None
    overlap: dict[str, float] = {}
    if topk_ref is not None:
        ref_set = topk_ref  # [B, Sq, topk]
        total = ref_set.numel()

        def _overlap(cand: torch.Tensor) -> float:
            if cand.shape != ref_set.shape:
                return float("nan")
            ref_sorted, _ = ref_set.sort(dim=-1)
            cand_sorted, _ = cand.sort(dim=-1)
            # Proper set overlap per row: intersection count / topk
            topk_n = ref_set.size(-1)
            B, Sq, _ = ref_set.shape
            match = 0
            for b in range(B):
                for s in range(Sq):
                    rs = set(ref_set[b, s].tolist())
                    cs = set(cand[b, s].tolist())
                    match += len(rs & cs)
            return 100.0 * match / (B * Sq * topk_n)

        if "FP8" in per_variant and "topk_fp8" in per_variant["FP8"]:
            overlap["FP8"] = _overlap(per_variant["FP8"]["topk_fp8"])
        if "FP4" in per_variant and "topk_fp4" in per_variant["FP4"]:
            overlap["FP4"] = _overlap(per_variant["FP4"]["topk_fp4"])
        overlap["BF16"] = 100.0  # trivially
    results["topk_overlap_vs_bf16_pct"] = overlap

    # Strip tensor objects before returning (Modal serializes with cloudpickle).
    def _scrub(d: Any) -> Any:
        if isinstance(d, dict):
            return {k: _scrub(v) for k, v in d.items() if not isinstance(v, torch.Tensor)}
        return d

    results["variants"] = {k: _scrub(v) for k, v in per_variant.items()}

    print("=" * 72)
    print("[bench] RESULTS")
    print("=" * 72)
    for name, data in results["variants"].items():
        print(f"-- {name} --")
        for k, v in data.items():
            print(f"   {k}: {v}")
    print(f"[bench] topk overlap vs BF16: {overlap}")

    return results


@app.local_entrypoint()
def main() -> None:
    """Run the DSA indexer bench on Modal B200 and pretty-print results."""
    print("=== cppmega Modal DSA indexer bench (Stream F, task #82) ===")
    print(f"GPU spec: {_GPU_SPEC}")
    res = bench.remote()
    print()
    print("=" * 72)
    print("FINAL RESULTS")
    print("=" * 72)
    print(f"device:          {res.get('cuda_device')}")
    print(f"cuda cap:        {res.get('cuda_cap')}")
    print(f"torch:           {res.get('torch_version')} (cuda {res.get('torch_cuda')})")
    print(f"TE available:    {res.get('te_available')} version={res.get('te_version')}")
    print(f"TE FP8 recipe:   {res.get('te_fp8_available')}")
    print(f"TE FP4 recipe:   {res.get('te_fp4_available')} name={res.get('te_fp4_recipe_name')}")
    print(f"shape:           {res.get('shape')}")
    print()
    variants = res.get("variants", {})
    for name in ("BF16", "FP8", "FP4"):
        data = variants.get(name, {})
        print(f"--- {name} ---")
        if not data or "status" in data and "error" in str(data.get("status", "")):
            print(f"   {data}")
            continue
        if "linear_fwd" in data:
            print(f"  linear_fwd:        mean={data['linear_fwd']['mean_us']:.1f} µs "
                  f"p50={data['linear_fwd']['p50_us']:.1f} p99={data['linear_fwd']['p99_us']:.1f}")
            print(f"  linear_bwd:        mean={data['linear_bwd']['mean_us']:.1f} µs "
                  f"p50={data['linear_bwd']['p50_us']:.1f} p99={data['linear_bwd']['p99_us']:.1f}")
            print(f"  index_compute_fwd: mean={data['index_compute_fwd']['mean_us']:.1f} µs "
                  f"p50={data['index_compute_fwd']['p50_us']:.1f} p99={data['index_compute_fwd']['p99_us']:.1f}")
            print(f"  index_compute_bwd: mean={data['index_compute_bwd']['mean_us']:.1f} µs "
                  f"p50={data['index_compute_bwd']['p50_us']:.1f} p99={data['index_compute_bwd']['p99_us']:.1f}")
            print(f"  fused_qk_topk:     mean={data['fused_qk_topk']['mean_us']:.1f} µs "
                  f"p50={data['fused_qk_topk']['p50_us']:.1f} p99={data['fused_qk_topk']['p99_us']:.1f}")
            print(f"  peak_memory_MB:    {data['peak_memory_MB']:.1f}")
        else:
            print(f"  status: {data}")
    print()
    print(f"topk overlap vs BF16: {res.get('topk_overlap_vs_bf16_pct')}")
    print()
    print("=" * 72)
    print("Remember to copy this output into docs/blackwell_feature_sweep_2026_04_12.md")
    print("under the 'DSA indexer FP4/FP8/BF16 micro-bench on Modal B200' section.")
    print("=" * 72)
