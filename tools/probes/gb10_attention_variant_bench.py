"""Benchmark GB10 attention backend candidates.

This probe intentionally separates training attention from decode-only MLA
paths.  FlashInfer XQA MLA is a paged decode kernel: it has no backward path and
cannot be used for 100-step training loss without a separate training backend.
TE fused/FA4 and PyTorch SDPA are measured with fwd+bwd.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BenchResult:
    name: str
    mode: str
    status: str
    median_ms: float | None = None
    tokens_per_sec: float | None = None
    max_alloc_gib: float | None = None
    output_dtype: str | None = None
    note: str | None = None
    error: str | None = None


def _sync() -> None:
    torch.cuda.synchronize()


def _time_cuda(fn: Callable[[], Any], *, warmup: int, repeat: int) -> tuple[list[float], Any]:
    out = None
    for _ in range(warmup):
        out = fn()
    _sync()
    samples: list[float] = []
    for _ in range(repeat):
        start = time.perf_counter()
        out = fn()
        _sync()
        samples.append((time.perf_counter() - start) * 1000.0)
    return samples, out


def _result_from_samples(
    name: str,
    mode: str,
    samples: list[float],
    *,
    token_count: int,
    output: torch.Tensor,
    note: str | None = None,
) -> BenchResult:
    median_ms = statistics.median(samples)
    return BenchResult(
        name=name,
        mode=mode,
        status="pass",
        median_ms=median_ms,
        tokens_per_sec=token_count / (median_ms / 1000.0),
        max_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
        output_dtype=str(output.dtype),
        note=note,
    )


def _bench_te_dpa(
    *,
    backend: str,
    seqlen: int,
    batch_size: int,
    num_heads: int,
    num_gqa_groups: int,
    qk_dim: int,
    v_dim: int,
    warmup: int,
    repeat: int,
) -> BenchResult:
    os.environ["NVTE_FLASH_ATTN"] = "1" if backend == "flash" else "0"
    os.environ["NVTE_FUSED_ATTN"] = "1" if backend == "fused" else "0"
    os.environ["NVTE_UNFUSED_ATTN"] = "1" if backend == "unfused" else "0"

    from transformer_engine.pytorch.attention import DotProductAttention

    torch.manual_seed(1234)
    q = torch.randn(
        seqlen,
        batch_size,
        num_heads,
        qk_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        seqlen,
        batch_size,
        num_gqa_groups,
        qk_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        seqlen,
        batch_size,
        num_gqa_groups,
        v_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    dpa = DotProductAttention(
        num_attention_heads=num_heads,
        kv_channels=(qk_dim, v_dim),
        num_gqa_groups=num_gqa_groups,
        qkv_format="sbhd",
        attention_dropout=0.0,
        attn_mask_type="causal",
    )

    def run() -> torch.Tensor:
        for tensor in (q, k, v):
            tensor.grad = None
        out = dpa(
            q,
            k,
            v,
            qkv_format="sbhd",
            max_seqlen_q=seqlen,
            max_seqlen_kv=seqlen,
            attn_mask_type="causal",
            core_attention_bias_type="no_bias",
        )
        out.float().sum().backward()
        return out

    torch.cuda.reset_peak_memory_stats()
    samples, out = _time_cuda(run, warmup=warmup, repeat=repeat)
    return _result_from_samples(
        f"te_{backend}",
        "train_fwd_bwd",
        samples,
        token_count=seqlen * batch_size,
        output=out,
        note="TE DotProductAttention fwd+bwd",
    )


def _bench_torch_sdpa(
    *,
    seqlen: int,
    batch_size: int,
    num_heads: int,
    qk_dim: int,
    v_dim: int,
    warmup: int,
    repeat: int,
) -> BenchResult:
    import torch.nn.functional as F

    torch.manual_seed(1234)
    q = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        qk_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        qk_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        num_heads,
        seqlen,
        v_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def run() -> torch.Tensor:
        for tensor in (q, k, v):
            tensor.grad = None
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out.float().sum().backward()
        return out

    torch.cuda.reset_peak_memory_stats()
    samples, out = _time_cuda(run, warmup=warmup, repeat=repeat)
    return _result_from_samples(
        "torch_sdpa_default",
        "train_fwd_bwd",
        samples,
        token_count=seqlen * batch_size,
        output=out,
        note="torch.nn.functional.scaled_dot_product_attention fwd+bwd",
    )


def _bench_flashinfer_xqa(
    *,
    batch_size: int,
    seq_len: int,
    page_size: int,
    backend: str,
    warmup: int,
    repeat: int,
) -> BenchResult:
    import flashinfer

    torch.manual_seed(1234)
    num_q_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    q = torch.randn(
        batch_size,
        1,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    num_pages = batch_size * ((seq_len + page_size - 1) // page_size)
    kv_cache = torch.randn(
        num_pages,
        page_size,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    ).to(torch.float8_e4m3fn)
    pages_per_seq = num_pages // batch_size
    block_tables = torch.arange(num_pages, device="cuda", dtype=torch.int32).reshape(
        batch_size, pages_per_seq
    )
    seq_lens = torch.full((batch_size,), seq_len, device="cuda", dtype=torch.int32)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    def run() -> torch.Tensor:
        return flashinfer.decode.trtllm_batch_decode_with_kv_cache_mla(
            query=q,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace,
            qk_nope_head_dim=128,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len,
            bmm1_scale=1.0 / (head_dim**0.5),
            bmm2_scale=1.0,
            backend=backend,
        )

    torch.cuda.reset_peak_memory_stats()
    samples, out = _time_cuda(run, warmup=warmup, repeat=repeat)
    return _result_from_samples(
        f"flashinfer_mla_decode_{backend}",
        "decode_fwd_only",
        samples,
        token_count=batch_size,
        output=out,
        note="FlashInfer XQA/trtllm MLA decode has no backward/training loss path",
    )


def _run_one(name: str, fn: Callable[[], BenchResult]) -> BenchResult:
    try:
        return fn()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return BenchResult(
            name=name,
            mode="unknown",
            status="fail",
            error=f"{type(exc).__name__}: {exc}",
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seqlen", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--heads", type=int, default=28)
    parser.add_argument("--gqa-groups", type=int, default=28)
    parser.add_argument("--qk-dim", type=int, default=128)
    parser.add_argument("--v-dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--flashinfer-batch-size", type=int, default=1)
    parser.add_argument("--flashinfer-seq-len", type=int, default=4096)
    parser.add_argument("--flashinfer-page-size", type=int, default=64)
    parser.add_argument(
        "--variant",
        action="append",
        choices=(
            "te_fused",
            "te_flash",
            "te_unfused",
            "torch_sdpa",
            "flashinfer_xqa",
            "flashinfer_auto",
            "flashinfer_trtllm",
            "flashinfer_cute",
        ),
        help="Variant to run; defaults to all supported candidates.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    torch.cuda.set_device(0)
    variants = args.variant or [
        "te_fused",
        "te_flash",
        "te_unfused",
        "torch_sdpa",
        "flashinfer_xqa",
        "flashinfer_auto",
        "flashinfer_trtllm",
        "flashinfer_cute",
    ]

    def te_runner(backend: str) -> Callable[[], BenchResult]:
        return lambda: _bench_te_dpa(
            backend=backend,
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            num_heads=args.heads,
            num_gqa_groups=args.gqa_groups,
            qk_dim=args.qk_dim,
            v_dim=args.v_dim,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    def flashinfer_runner(backend: str) -> Callable[[], BenchResult]:
        return lambda: _bench_flashinfer_xqa(
            batch_size=args.flashinfer_batch_size,
            seq_len=args.flashinfer_seq_len,
            page_size=args.flashinfer_page_size,
            backend=backend,
            warmup=args.warmup,
            repeat=args.repeat,
        )

    runners: dict[str, Callable[[], BenchResult]] = {
        "te_fused": te_runner("fused"),
        "te_flash": te_runner("flash"),
        "te_unfused": te_runner("unfused"),
        "torch_sdpa": lambda: _bench_torch_sdpa(
            seqlen=args.seqlen,
            batch_size=args.batch_size,
            num_heads=args.heads,
            qk_dim=args.qk_dim,
            v_dim=args.v_dim,
            warmup=args.warmup,
            repeat=args.repeat,
        ),
        "flashinfer_xqa": flashinfer_runner("xqa"),
        "flashinfer_auto": flashinfer_runner("auto"),
        "flashinfer_trtllm": flashinfer_runner("trtllm-gen"),
        "flashinfer_cute": flashinfer_runner("cute-dsl"),
    }
    results = [_run_one(name, runners[name]) for name in variants]
    payload = {
        "device": torch.cuda.get_device_name(0),
        "capability": torch.cuda.get_device_capability(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "args": vars(args),
        "results": [asdict(result) for result in results],
        "pythonpath": sys.path[:6],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
