"""Focused LinearCE correctness and timing probe.

This is intentionally smaller than an end-to-end training run.  It exercises
the main-head + MTP shared-weight pattern with CCE reduction="sum" masked
labels, then times CCE and optionally Liger on one synthetic CE shape.

The ``--include-main-mtp`` mode measures the launch-count question directly:
main head plus N MTP heads as separate CCE calls versus one concatenated CCE
call matching ``CPPMEGA_CCE_FUSE_MAIN_MTP_CE=1``.
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


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _median(values: list[float]) -> float:
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _masked_mtp_labels(
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    mtp_depth: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    mtp_labels = labels.clone()
    mtp_mask = loss_mask.clone()
    masked_labels: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for _ in range(mtp_depth):
        mtp_labels = torch.roll(mtp_labels, shifts=-1, dims=-1)
        mtp_labels[:, -1] = 0
        mtp_mask = torch.roll(mtp_mask, shifts=-1, dims=-1)
        mtp_mask[:, -1] = 0
        masked_labels.append(
            torch.where(
                mtp_mask.bool(),
                mtp_labels,
                torch.full_like(mtp_labels, IGNORE_INDEX),
            )
        )
        masks.append(mtp_mask.clone())
    return masked_labels, masks


def run_main_mtp_call_count_timing(args: argparse.Namespace, filter_eps: Any) -> None:
    """Time separate main+MTP CCE calls against one concatenated CCE call."""

    torch.manual_seed(20260501)
    device = "cuda"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    batch, seq, hidden, vocab, mtp_depth = (
        args.batch,
        args.seq,
        args.hidden,
        args.vocab,
        args.mtp_depth,
    )
    total_seq = (1 + mtp_depth) * seq

    hidden_base = torch.randn(total_seq, batch, hidden, device=device, dtype=dtype) * 0.2
    weight_base = torch.randn(vocab, hidden, device=device, dtype=dtype) * 0.02
    labels = torch.randint(0, vocab, (batch, seq), device=device)
    loss_mask = (torch.rand(batch, seq, device=device) > args.mask_prob).float()
    mtp_labels, mtp_masks = _masked_mtp_labels(labels, loss_mask, mtp_depth)
    fused_labels = torch.cat([labels, *mtp_labels], dim=-1)

    def _flatten_targets(targets: torch.Tensor) -> torch.Tensor:
        return targets.transpose(0, 1).contiguous().reshape(-1)

    labels_1d = _flatten_targets(labels)
    mtp_labels_1d = [_flatten_targets(targets) for targets in mtp_labels]
    fused_labels_1d = _flatten_targets(fused_labels)
    main_mask_1d = loss_mask.transpose(0, 1).contiguous().reshape(-1)
    mtp_masks_1d = [mask.transpose(0, 1).contiguous().reshape(-1) for mask in mtp_masks]

    mtp_scale = args.mtp_loss_scale / max(mtp_depth, 1)

    def separate_calls() -> None:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)

        total = cce_linear_cross_entropy(
            hidden_t[:seq].contiguous().reshape(seq * batch, hidden),
            weight_t,
            labels_1d,
            reduction="none",
            ignore_index=IGNORE_INDEX,
            filter_eps=filter_eps,
        )
        loss = (total * main_mask_1d).sum()
        for depth_idx in range(mtp_depth):
            mtp_sum = cce_linear_cross_entropy(
                hidden_t[(depth_idx + 1) * seq : (depth_idx + 2) * seq]
                .contiguous()
                .reshape(seq * batch, hidden),
                weight_t,
                mtp_labels_1d[depth_idx],
                reduction="sum",
                ignore_index=IGNORE_INDEX,
                filter_eps=filter_eps,
            )
            num_tokens = mtp_masks_1d[depth_idx].sum().clamp(min=1)
            loss = loss + mtp_scale * mtp_sum / num_tokens
        loss.backward()

    def fused_one_call() -> None:
        _run_fused_one_call()

    def _run_fused_one_call() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)
        fused_loss = cce_linear_cross_entropy(
            hidden_t.contiguous().reshape(total_seq * batch, hidden),
            weight_t,
            fused_labels_1d,
            reduction="none",
            ignore_index=IGNORE_INDEX,
            filter_eps=filter_eps,
        )
        fused_loss = fused_loss.reshape(total_seq, batch).transpose(0, 1).contiguous()
        main_loss = (fused_loss[:, :seq] * loss_mask).sum()
        mtp_loss = fused_loss[:, seq:].reshape(batch, mtp_depth, seq)
        loss = main_loss
        for depth_idx in range(mtp_depth):
            mtp_sum = mtp_loss[:, depth_idx, :].sum()
            num_tokens = mtp_masks[depth_idx].sum().clamp(min=1)
            loss = loss + mtp_scale * mtp_sum / num_tokens
        loss.backward()
        return loss.detach(), hidden_t.grad.detach(), weight_t.grad.detach()

    def _run_separate_calls() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_t = hidden_base.detach().clone().requires_grad_(True)
        weight_t = weight_base.detach().clone().requires_grad_(True)

        total = cce_linear_cross_entropy(
            hidden_t[:seq].contiguous().reshape(seq * batch, hidden),
            weight_t,
            labels_1d,
            reduction="none",
            ignore_index=IGNORE_INDEX,
            filter_eps=filter_eps,
        )
        loss = (total * main_mask_1d).sum()
        for depth_idx in range(mtp_depth):
            mtp_sum = cce_linear_cross_entropy(
                hidden_t[(depth_idx + 1) * seq : (depth_idx + 2) * seq]
                .contiguous()
                .reshape(seq * batch, hidden),
                weight_t,
                mtp_labels_1d[depth_idx],
                reduction="sum",
                ignore_index=IGNORE_INDEX,
                filter_eps=filter_eps,
            )
            num_tokens = mtp_masks_1d[depth_idx].sum().clamp(min=1)
            loss = loss + mtp_scale * mtp_sum / num_tokens
        loss.backward()
        return loss.detach(), hidden_t.grad.detach(), weight_t.grad.detach()

    print(
        "[main_mtp] "
        f"batch={batch} seq={seq} mtp_depth={mtp_depth} hidden={hidden} "
        f"vocab={vocab} dtype={dtype} filter_eps={filter_eps!r}"
    )
    if args.check_main_mtp_correctness:
        separate_loss, separate_hidden_grad, separate_weight_grad = _run_separate_calls()
        fused_loss, fused_hidden_grad, fused_weight_grad = _run_fused_one_call()
        torch.cuda.synchronize()
        loss_rel = abs(fused_loss.item() - separate_loss.item()) / max(
            abs(separate_loss.item()), 1.0
        )
        print(
            "  correctness: "
            f"loss_rel={loss_rel:.3e} "
            f"grad_hidden_rel_l2={_rel_l2(fused_hidden_grad, separate_hidden_grad):.3e} "
            f"grad_weight_rel_l2={_rel_l2(fused_weight_grad, separate_weight_grad):.3e}"
        )
    results: dict[str, tuple[list[float], float]] = {}
    for name, fn in (
        ("separate_main_plus_mtp_calls", separate_calls),
        ("fused_one_cce_call", fused_one_call),
    ):
        times, peak_gib = _time_cuda(fn, args.iters)
        results[name] = (times, peak_gib)
        print(
            f"  {name}: avg_ms={_mean(times):.2f} median_ms={_median(times):.2f} "
            f"times_ms={[round(t, 2) for t in times]} peak_gib={peak_gib:.3f}"
        )

    separate_avg = _mean(results["separate_main_plus_mtp_calls"][0])
    fused_avg = _mean(results["fused_one_cce_call"][0])
    print(
        "  fused_vs_separate: "
        f"delta_ms={fused_avg - separate_avg:.2f} speedup={separate_avg / fused_avg:.3f}x"
    )


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
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--seq", type=int, default=2048)
    parser.add_argument("--mtp-depth", type=int, default=2)
    parser.add_argument("--hidden", type=int, default=3584)
    parser.add_argument("--vocab", type=int, default=65536)
    parser.add_argument("--dtype", choices=("bf16", "fp16"), default="bf16")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--filter-eps", default="high")
    parser.add_argument("--mask-prob", type=float, default=0.1)
    parser.add_argument("--mtp-loss-scale", type=float, default=0.1)
    parser.add_argument("--include-main-mtp", action="store_true")
    parser.add_argument("--check-main-mtp-correctness", action="store_true")
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
    filter_eps = _filter_eps(args.filter_eps)
    if args.include_main_mtp:
        run_main_mtp_call_count_timing(args, filter_eps)
    run_cce_timing(args, _filter_eps(args.filter_eps))
    if args.include_liger:
        run_liger_timing(args)


if __name__ == "__main__":
    main()
