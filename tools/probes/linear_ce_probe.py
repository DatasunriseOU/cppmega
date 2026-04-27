"""Focused LinearCE correctness and timing probe.

This is intentionally smaller than an end-to-end training run.  It exercises
the main-head + MTP shared-weight pattern with CCE reduction="sum" masked
labels, then times CCE and optionally Liger on one synthetic CE shape.
"""

from __future__ import annotations

import argparse
import importlib.util
from typing import Any

import torch
import torch.nn.functional as F
from cut_cross_entropy import linear_cross_entropy as cce_linear_cross_entropy


IGNORE_INDEX = -100


def _filter_eps(value: str) -> str | float | None:
    value = value.lower()
    if value in ("none", "off", "false", "0"):
        return None
    if value in ("auto", "high"):
        return value
    return float(value)


def _rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    diff = a.float() - b.float()
    return diff.norm().item() / max(a.float().norm().item(), 1e-12)


def _time_cuda(fn, iters: int) -> tuple[list[float], float]:
    for _ in range(1):
        fn()
    torch.cuda.synchronize()

    times: list[float] = []
    peak_gib = 0.0
    for _ in range(iters):
        torch.cuda.reset_peak_memory_stats()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
        peak_gib = max(peak_gib, torch.cuda.max_memory_allocated() / 2**30)
    return times, peak_gib


def run_shared_weight_correctness(dtype: torch.dtype) -> None:
    torch.manual_seed(777)
    device = "cuda"
    tokens, hidden, vocab = 256, 256, 4096
    calls = 3

    hiddens = [
        torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.2 for _ in range(calls)
    ]
    weight = torch.randn(vocab, hidden, device=device, dtype=dtype) * 0.02
    targets = [torch.randint(0, vocab, (tokens,), device=device) for _ in range(calls)]
    masks = [(torch.rand(tokens, device=device) > 0.1).float() for _ in range(calls)]
    masked_targets = [
        torch.where(mask.bool(), target, target.new_full(target.shape, IGNORE_INDEX))
        for target, mask in zip(targets, masks)
    ]
    scales = [1.0, 0.15, 0.15]

    ref_h = [hidden.detach().float().requires_grad_(True) for hidden in hiddens]
    ref_w = weight.detach().float().requires_grad_(True)
    ref_loss = sum(
        scales[i]
        * (
            F.cross_entropy(F.linear(ref_h[i], ref_w), targets[i], reduction="none")
            * masks[i]
        ).sum()
        for i in range(calls)
    )
    ref_loss.backward()

    cce_h = [hidden.detach().clone().requires_grad_(True) for hidden in hiddens]
    cce_w = weight.detach().clone().requires_grad_(True)
    cce_loss = sum(
        scales[i]
        * cce_linear_cross_entropy(
            cce_h[i],
            cce_w,
            masked_targets[i],
            reduction="sum",
            ignore_index=IGNORE_INDEX,
            filter_eps=None,
        )
        for i in range(calls)
    )
    cce_loss.backward()
    torch.cuda.synchronize()

    print("[correctness] CCE shared-weight 3-call reduction=sum")
    print(
        f"  loss_rel={abs(cce_loss.item() - ref_loss.item()) / max(abs(ref_loss.item()), 1.0):.3e}"
    )
    print(f"  grad_w_rel_l2={_rel_l2(cce_w.grad, ref_w.grad):.3e}")
    for i in range(calls):
        print(f"  grad_h{i}_rel_l2={_rel_l2(cce_h[i].grad, ref_h[i].grad):.3e}")
    print(
        "  finite="
        f"{torch.isfinite(cce_w.grad).all().item() and all(torch.isfinite(h.grad).all().item() for h in cce_h)}"
    )


def run_cce_timing(args: argparse.Namespace, filter_eps: Any) -> None:
    torch.manual_seed(456)
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokens, hidden, vocab = args.tokens, args.hidden, args.vocab

    hidden_base = torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.2
    weight_base = torch.randn(vocab, hidden, device=device, dtype=dtype) * 0.02
    targets = torch.randint(0, vocab, (tokens,), device=device)
    mask = (torch.rand(tokens, device=device) > 0.1).float()
    masked_targets = torch.where(mask.bool(), targets, targets.new_full(targets.shape, IGNORE_INDEX))

    def cce_none_mask() -> None:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)
        loss = cce_linear_cross_entropy(
            hidden_t,
            weight_t,
            targets,
            reduction="none",
            ignore_index=IGNORE_INDEX,
            filter_eps=None,
        )
        (loss * mask).sum().backward()

    def cce_sum_masked() -> None:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)
        loss = cce_linear_cross_entropy(
            hidden_t,
            weight_t,
            masked_targets,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
            filter_eps=None,
        )
        loss.backward()

    def cce_sum_filter() -> None:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)
        loss = cce_linear_cross_entropy(
            hidden_t,
            weight_t,
            masked_targets,
            reduction="sum",
            ignore_index=IGNORE_INDEX,
            filter_eps=filter_eps,
        )
        loss.backward()

    print(f"[timing] shape tokens={tokens} hidden={hidden} vocab={vocab} dtype={dtype}")
    for name, fn in (
        ("cce_none_mask_filter_none", cce_none_mask),
        ("cce_sum_masked_filter_none", cce_sum_masked),
        (f"cce_sum_masked_filter_{args.filter_eps}", cce_sum_filter),
    ):
        times, peak_gib = _time_cuda(fn, args.iters)
        print(
            f"  {name}: avg_ms={sum(times) / len(times):.2f} "
            f"times_ms={[round(t, 2) for t in times]} peak_gib={peak_gib:.3f}"
        )


def run_liger_timing(args: argparse.Namespace) -> None:
    if importlib.util.find_spec("liger_kernel") is None:
        print("[timing] liger skipped: liger_kernel not installed")
        return

    from liger_kernel.ops.fused_linear_cross_entropy import (
        LigerFusedLinearCrossEntropyFunction,
    )

    torch.manual_seed(789)
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    tokens, hidden, vocab = args.tokens, args.hidden, args.vocab

    hidden_base = torch.randn(tokens, hidden, device=device, dtype=dtype) * 0.2
    weight_base = torch.randn(vocab, hidden, device=device, dtype=dtype) * 0.02
    targets = torch.randint(0, vocab, (tokens,), device=device)
    mask = (torch.rand(tokens, device=device) > 0.1).float()
    masked_targets = torch.where(mask.bool(), targets, targets.new_full(targets.shape, IGNORE_INDEX))

    def liger_mean_broadcast() -> None:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)
        loss, _, _ = LigerFusedLinearCrossEntropyFunction.apply(
            hidden_t,
            weight_t,
            masked_targets,
            None,
            None,
            IGNORE_INDEX,
            0.0,
            0.0,
            "mean",
            None,
            False,
        )
        (loss * mask.sum()).backward()

    times, peak_gib = _time_cuda(liger_mean_broadcast, args.iters)
    print(
        f"  liger_mean_broadcast: avg_ms={sum(times) / len(times):.2f} "
        f"times_ms={[round(t, 2) for t in times]} peak_gib={peak_gib:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--filter-eps", default="high")
    parser.add_argument("--include-liger", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("linear_ce_probe requires CUDA")

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    print(
        f"[env] torch={torch.__version__} device={torch.cuda.get_device_name()} "
        f"cc={torch.cuda.get_device_capability()} dtype={dtype}"
    )
    run_shared_weight_correctness(dtype)
    run_cce_timing(args, _filter_eps(args.filter_eps))
    if args.include_liger:
        run_liger_timing(args)


if __name__ == "__main__":
    main()
