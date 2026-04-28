"""End-to-end forward+backward microbench for the three M²RNN kernels at
NAM56R-like dimensions.

We instantiate the production ``CppMegaM2RNNMixer`` (Option B in the task
brief) and route through the three available kernels via the
``CPPMEGA_M2RNN_KERNEL`` env var:

  * ``triton``   -- the existing sequential Triton scan (default in prod)
  * ``torch``    -- the pure-PyTorch reference loop (debug/parity)
  * ``pararnn``  -- the new ParaRNN-style Newton + Brent-Kung parallel scan
                    (Phase C; auto-mode picks fp32 Triton kernels on CUDA)

For each kernel we record:

  * Mean wall time (ms) over 5 timed iters of fwd+loss+bwd
  * Peak GPU memory (MiB) during a single iter
  * For pararnn: max-abs gradient delta vs the triton path on the input
    projection weight (this is the easiest single-tensor parity proxy;
    the full set of params would be noisier and not actionable).

Run:  python tools/bench_m2rnn_kernels_nam56r.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch


# ---------------------------------------------------------------------------
# TP=1 distributed init (TENorm / Megatron expects a process group)
# ---------------------------------------------------------------------------


def _init_tp1() -> None:
    import torch.distributed as dist

    if dist.is_initialized():
        return
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", rank=0, world_size=1)

    from megatron.core import parallel_state

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1, pipeline_model_parallel_size=1,
        )


# ---------------------------------------------------------------------------
# Mixer build
# ---------------------------------------------------------------------------


@dataclass
class BenchShape:
    batch: int = 2
    seq: int = 4096
    hidden_size: int = 3520  # 44 heads at (k=64, v=16) -> 44 * 80
    num_attention_heads: int = 44


def _build_mixer(shape: BenchShape):
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=shape.hidden_size,
        num_attention_heads=shape.num_attention_heads,
        num_query_groups=shape.num_attention_heads,
        ffn_hidden_size=shape.hidden_size * 4,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        params_dtype=torch.bfloat16,
        bf16=True,
        use_cpu_initialization=False,
    )

    from cppmega.megatron.m2rnn_spec import CppMegaM2RNNMixer

    torch.manual_seed(20260428)
    torch.cuda.manual_seed_all(20260428)
    mixer = CppMegaM2RNNMixer(
        config=config, d_model=shape.hidden_size,
    ).cuda()
    return mixer


# ---------------------------------------------------------------------------
# Bench helpers
# ---------------------------------------------------------------------------


def _make_input(shape: BenchShape, dtype: torch.dtype, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        shape.seq, shape.batch, shape.hidden_size,
        generator=g, device="cuda", dtype=dtype,
    )
    x.requires_grad_(True)
    return x


def _zero_grads(mixer) -> None:
    for p in mixer.parameters():
        if p.grad is not None:
            p.grad = None


@dataclass
class BenchResult:
    kernel: str
    mean_ms: float
    stdev_ms: float
    peak_mib: float
    out_finite: bool
    grad_finite: bool
    fwd_only_ms: Optional[float] = None
    grad_delta_vs_triton: Optional[float] = None


def _run_kernel(
    kernel: str,
    shape: BenchShape,
    n_warmup: int = 2,
    n_iters: int = 5,
) -> tuple[BenchResult, dict[str, torch.Tensor]]:
    """Run warmup + timed iters for one kernel, returning the bench result and
    a dict of {param_name: param.grad.detach().clone()} for parity comparison.
    """
    os.environ["CPPMEGA_M2RNN_KERNEL"] = kernel

    mixer = _build_mixer(shape)
    x = _make_input(shape, dtype=torch.bfloat16, seed=42)

    # Warmup.
    for _ in range(n_warmup):
        _zero_grads(mixer)
        if x.grad is not None:
            x.grad = None
        out, _ = mixer(x)
        loss = out.float().pow(2).mean()
        loss.backward()
    torch.cuda.synchronize()

    # Time fwd+bwd.
    torch.cuda.reset_peak_memory_stats()
    times_ms: list[float] = []
    fwd_only_ms: list[float] = []
    out_finite = True
    for _ in range(n_iters):
        _zero_grads(mixer)
        if x.grad is not None:
            x.grad = None

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out, _ = mixer(x)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        loss = out.float().pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        times_ms.append((t2 - t0) * 1e3)
        fwd_only_ms.append((t1 - t0) * 1e3)
        out_finite = out_finite and bool(torch.isfinite(out).all().item())

    peak_bytes = torch.cuda.max_memory_allocated()
    grad_finite = bool(torch.isfinite(x.grad).all().item()) if x.grad is not None else True

    # Capture gradient on input_projection.weight as the parity proxy: it's a
    # single dense Linear weight, sees gradient through every kernel via the
    # same path (q/k/v/g all derive from this projection), and is large enough
    # that fp32->bf16 roundoff is well below the Newton-residual scale.
    grads = {
        name: p.grad.detach().clone()
        for name, p in mixer.named_parameters()
        if p.grad is not None and "input_projection" in name
    }

    res = BenchResult(
        kernel=kernel,
        mean_ms=statistics.mean(times_ms),
        stdev_ms=statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        peak_mib=peak_bytes / (1024 * 1024),
        out_finite=out_finite,
        grad_finite=grad_finite,
        fwd_only_ms=statistics.mean(fwd_only_ms),
    )

    # Free the mixer + activations so the next kernel starts clean.
    del mixer, x, out, loss
    torch.cuda.empty_cache()

    return res, grads


# ---------------------------------------------------------------------------
# Pretty-print
# ---------------------------------------------------------------------------


def _format_table(results: list[BenchResult]) -> str:
    headers = [
        "kernel", "fwd+bwd ms (mean)", "+/- stdev", "fwd only (ms)",
        "peak GPU (MiB)", "out finite", "grad finite",
        "max|grad-grad_triton| (input_proj)",
    ]

    rows = []
    for r in results:
        rows.append([
            r.kernel,
            f"{r.mean_ms:.2f}",
            f"{r.stdev_ms:.2f}",
            f"{r.fwd_only_ms:.2f}" if r.fwd_only_ms is not None else "-",
            f"{r.peak_mib:.1f}",
            "yes" if r.out_finite else "NO",
            "yes" if r.grad_finite else "NO",
            (f"{r.grad_delta_vs_triton:.3e}"
             if r.grad_delta_vs_triton is not None else "-"),
        ])

    widths = [
        max(len(h), max(len(row[i]) for row in rows)) for i, h in enumerate(headers)
    ]

    def _fmt_row(cells):
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [sep, _fmt_row(headers), sep]
    for row in rows:
        lines.append(_fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=3520)
    parser.add_argument("--num-heads", type=int, default=44)
    parser.add_argument(
        "--kernels", nargs="+",
        default=["triton", "torch", "pararnn"],
        choices=["triton", "torch", "pararnn"],
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA not available; bench requires a GPU.", file=sys.stderr)
        return 1

    _init_tp1()

    shape = BenchShape(
        batch=args.batch,
        seq=args.seq,
        hidden_size=args.hidden,
        num_attention_heads=args.num_heads,
    )
    print(f"Hardware:  {torch.cuda.get_device_name(0)}")
    print(
        f"Shape:     B={shape.batch}, S={shape.seq}, "
        f"hidden={shape.hidden_size}, heads={shape.num_attention_heads} "
        f"(k_head_dim=64, v_head_dim=16, dtype=bf16)"
    )
    print(f"Iters:     warmup={args.warmup}, timed={args.iters}")
    print()

    results: list[BenchResult] = []
    triton_grads: Optional[dict[str, torch.Tensor]] = None
    for kernel in args.kernels:
        try:
            res, grads = _run_kernel(
                kernel, shape, n_warmup=args.warmup, n_iters=args.iters,
            )
        except Exception as e:  # surface failures in the table rather than die
            print(f"[{kernel}] FAILED: {e}", file=sys.stderr)
            res = BenchResult(
                kernel=kernel, mean_ms=float("nan"), stdev_ms=float("nan"),
                peak_mib=float("nan"), out_finite=False, grad_finite=False,
            )
            grads = {}
        else:
            print(
                f"[{kernel}] mean fwd+bwd: {res.mean_ms:6.2f} ms "
                f"(+/- {res.stdev_ms:.2f}); peak {res.peak_mib:.1f} MiB"
            )

        if kernel == "triton":
            triton_grads = grads
        elif kernel == "pararnn" and triton_grads is not None:
            # Compute parity on input_projection.weight (the canonical proxy).
            key = next(
                (k for k in grads if "input_projection.weight" in k), None,
            )
            if key is not None and key in triton_grads:
                a = grads[key].float()
                b = triton_grads[key].float()
                res.grad_delta_vs_triton = (a - b).abs().max().item()

        results.append(res)

    print()
    print(_format_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
