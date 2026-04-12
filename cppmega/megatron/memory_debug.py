"""CUDA memory inspection hook — enable via NANOCHAT_MEMORY_DEBUG=1."""
from __future__ import annotations
import os, traceback
import torch


def _fmt_bytes(b: int | float) -> str:
    if b >= 1 << 30: return f"{b / (1 << 30):.2f} GB"
    if b >= 1 << 20: return f"{b / (1 << 20):.2f} MB"
    if b >= 1 << 10: return f"{b / (1 << 10):.2f} KB"
    return f"{b} B"


def _print_memory_stats() -> None:
    """Print key fields from torch.cuda.memory_stats()."""
    stats = torch.cuda.memory_stats()
    fields = [
        ("allocated_bytes.all.current", "Allocated (current)"),
        ("allocated_bytes.all.peak", "Allocated (peak)"),
        ("reserved_bytes.all.current", "Reserved (current)"),
        ("reserved_bytes.all.peak", "Reserved (peak)"),
        ("active_bytes.all.current", "Active (current)"),
        ("active_bytes.all.peak", "Active (peak)"),
        ("num_alloc_retries", "Alloc retries (fragmentation)"),
        ("num_ooms", "OOM events"),
    ]
    print("\n===== CUDA Memory Stats =====")
    for key, label in fields:
        val = stats.get(key, 0)
        if "bytes" in key:
            print(f"  {label:40s}: {_fmt_bytes(val)}")
        else:
            print(f"  {label:40s}: {val}")
    inactive = stats.get("inactive_split_bytes.all.current", 0)
    if inactive > 0:
        print(f"  {'Inactive split (fragmentation)':40s}: {_fmt_bytes(inactive)}")


def _print_snapshot_top_allocations(top_n: int = 20) -> None:
    """Print top allocations from torch.cuda.memory_snapshot() grouped by size."""
    try:
        snapshot = torch.cuda.memory_snapshot()
    except Exception:
        print("  (memory_snapshot unavailable)")
        return
    if not snapshot:
        print("  (memory_snapshot empty — torch.cuda.memory._record_memory_history() "
              "must be called before forward pass)")
        return

    # Collect all active allocations from segments
    allocs: list[dict] = []
    for seg in snapshot:
        for block in seg.get("blocks", []):
            if block.get("state") == "active_allocated":
                allocs.append({"size": block.get("size", 0),
                               "frames": block.get("frames", [])})
    if not allocs:
        print("  (no active allocations in snapshot)")
        return

    allocs.sort(key=lambda a: a["size"], reverse=True)

    # Size buckets
    buckets: dict[str, int] = {">100MB": 0, ">10MB": 0, ">1MB": 0, "<1MB": 0}
    counts: dict[str, int] = {">100MB": 0, ">10MB": 0, ">1MB": 0, "<1MB": 0}
    for a in allocs:
        sz = a["size"]
        if sz > 100 << 20:   k = ">100MB"
        elif sz > 10 << 20:  k = ">10MB"
        elif sz > 1 << 20:   k = ">1MB"
        else:                 k = "<1MB"
        buckets[k] += sz; counts[k] += 1

    print(f"\n===== Allocation Size Buckets ({len(allocs)} active) =====")
    for b in [">100MB", ">10MB", ">1MB", "<1MB"]:
        print(f"  {b:10s}: {counts[b]:6d} allocs, {_fmt_bytes(buckets[b]):>10s} total")

    # Top N largest with stack traces
    print(f"\n===== Top {min(top_n, len(allocs))} Largest Allocations =====")
    skip_dirs = ("torch/nn/modules/module.py", "torch/_dynamo",
                 "torch/_inductor", "torch/autograd")
    for i, a in enumerate(allocs[:top_n]):
        print(f"\n  #{i+1}: {_fmt_bytes(a['size'])}")
        frames = a.get("frames", [])
        if not frames:
            print("    (no frame info)")
            continue
        shown = 0
        for f in frames:
            fn = f.get("filename", "")
            if any(s in fn for s in skip_dirs):
                continue
            print(f"    {fn}:{f.get('line', '?')} in {f.get('name', '?')}")
            shown += 1
            if shown >= 5:
                break
        if shown == 0:
            for f in frames[:3]:
                print(f"    {f.get('filename', '?')}:{f.get('line', '?')} in {f.get('name', '?')}")


def _print_param_memory(model: torch.nn.Module) -> None:
    """Print memory consumed by model parameters, gradients, and buffers."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = sum(p.grad.numel() * p.grad.element_size()
                     for p in model.parameters() if p.grad is not None)
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total = param_bytes + grad_bytes + buf_bytes
    residual = torch.cuda.memory_allocated() - total

    print("\n===== Model Memory =====")
    print(f"  {'Parameters':40s}: {_fmt_bytes(param_bytes)}")
    print(f"  {'Gradients':40s}: {_fmt_bytes(grad_bytes)}")
    print(f"  {'Buffers':40s}: {_fmt_bytes(buf_bytes)}")
    print(f"  {'Total (params+grads+bufs)':40s}: {_fmt_bytes(total)}")
    print(f"  {'Activations + optimizer + other':40s}: {_fmt_bytes(max(residual, 0))}")


def dump_memory_after_first_step(model: torch.nn.Module, step: int) -> None:
    """Call after step 0 forward+backward to dump detailed CUDA memory diagnostics.

    Only active when NANOCHAT_MEMORY_DEBUG=1, CUDA available, rank 0.
    """
    if step != 0 or not os.environ.get("NANOCHAT_MEMORY_DEBUG", ""):
        return
    if not torch.cuda.is_available():
        return
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    try:
        print("\n" + "=" * 60)
        print("  NANOCHAT_MEMORY_DEBUG: Post-step-0 memory inspection")
        print("=" * 60)
        _print_memory_stats()
        _print_param_memory(model)
        _print_snapshot_top_allocations(top_n=20)
        print("\n" + "=" * 60)
        print("  End memory inspection")
        print("=" * 60 + "\n")
        # Stop recording to avoid ongoing overhead
        try:
            torch.cuda.memory._record_memory_history(enabled=None)
        except Exception:
            pass
    except Exception as e:
        print(f"  [memory_debug] Error: {e}")
        traceback.print_exc()
