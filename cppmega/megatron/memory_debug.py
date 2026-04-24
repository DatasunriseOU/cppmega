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


def _collect_snapshot():
    try:
        return torch.cuda.memory._snapshot()
    except Exception:
        try:
            return {"segments": torch.cuda.memory_snapshot(), "device_traces": []}
        except Exception:
            return None


def _print_frames(frames: list[dict], max_frames: int = 5) -> None:
    skip_parts = (
        "memory_snapshot.cpp",
        "RegisterCUDA_",
        "RegisterBackendSelect.cpp",
        "torch/nn/modules/module.py",
        "torch/_dynamo",
        "torch/_inductor",
        "torch/autograd",
    )
    preferred = []
    for f in frames:
        fn = f.get("filename", "")
        if not fn or fn == "??" or any(s in fn for s in skip_parts):
            continue
        preferred.append(f)
    if not preferred:
        preferred = frames

    shown = 0
    for f in preferred:
        print(f"    {f.get('filename', '?')}:{f.get('line', '?')} in {f.get('name', '?')}")
        shown += 1
        if shown >= max_frames:
            break


def _print_snapshot_top_allocations(snapshot, top_n: int = 20) -> None:
    """Print top active allocations from a CUDA memory snapshot."""
    if snapshot is None:
        print("  (memory_snapshot unavailable)")
        return
    segments = snapshot.get("segments", snapshot) if isinstance(snapshot, dict) else snapshot
    if not segments:
        print("  (memory_snapshot empty — torch.cuda.memory._record_memory_history() "
              "must be called before forward pass)")
        return

    # Collect all active allocations from segments
    allocs: list[dict] = []
    for seg in segments:
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
    for i, a in enumerate(allocs[:top_n]):
        print(f"\n  #{i+1}: {_fmt_bytes(a['size'])}")
        frames = a.get("frames", [])
        if not frames:
            print("    (no frame info)")
            continue
        _print_frames(frames)


def _print_history_peak(snapshot, top_n: int = 12) -> None:
    """Replay alloc/free trace entries to identify transient peak allocations."""
    if not isinstance(snapshot, dict):
        return
    device_traces = snapshot.get("device_traces") or []
    if not device_traces:
        print("\n===== CUDA Allocation History =====")
        print("  (no device_traces in snapshot)")
        return

    print("\n===== CUDA Allocation History Peak =====")
    for device_idx, trace in enumerate(device_traces):
        current = 0
        peak = 0
        peak_index = -1
        active: dict[int, int] = {}
        alloc_events: list[dict] = []
        for idx, event in enumerate(trace):
            action = event.get("action")
            size = int(event.get("size", 0) or 0)
            addr = int(event.get("addr", 0) or 0)
            if action == "alloc":
                current += size
                if addr:
                    active[addr] = size
                alloc_events.append({**event, "_index": idx})
                if current > peak:
                    peak = current
                    peak_index = idx
            elif action == "free_completed":
                current -= active.pop(addr, size)
                if current < 0:
                    current = 0

        print(
            f"  device {device_idx}: {len(trace)} trace events, "
            f"traced_alloc_peak={_fmt_bytes(peak)}, traced_live_end={_fmt_bytes(current)}"
        )
        if peak_index >= 0:
            peak_event = trace[peak_index]
            print(
                f"  peak event #{peak_index}: {peak_event.get('action')} "
                f"{_fmt_bytes(int(peak_event.get('size', 0) or 0))}"
            )
            _print_frames(peak_event.get("frames", []), max_frames=6)

        alloc_events.sort(key=lambda e: int(e.get("size", 0) or 0), reverse=True)
        print(f"\n  Top {min(top_n, len(alloc_events))} allocation events:")
        for i, event in enumerate(alloc_events[:top_n], start=1):
            print(
                f"  #{i}: event={event.get('_index')} "
                f"size={_fmt_bytes(int(event.get('size', 0) or 0))} "
                f"action={event.get('action')}"
            )
            _print_frames(event.get("frames", []), max_frames=4)


def _call_or_value(value):
    return value() if callable(value) else value


def _storage_nbytes(tensor: torch.Tensor, seen_storages: set[int] | None = None) -> int:
    if seen_storages is not None:
        try:
            storage = tensor.untyped_storage()
            ptr = storage.data_ptr()
            if ptr in seen_storages:
                return 0
            seen_storages.add(ptr)
            return int(storage.nbytes())
        except Exception:
            pass
    return int(tensor.numel() * tensor.element_size())


def _object_nbytes(obj, seen_storages: set[int] | None = None) -> int:
    if obj is None:
        return 0
    if torch.is_tensor(obj):
        return _storage_nbytes(obj, seen_storages)

    # Megatron DDP exposes ParamAndGradBuffer objects through model.buffers.
    # Count their real tensor storage, not the integer metadata fields.
    total = 0
    for attr in ("param_data", "grad_data", "shared_buffer"):
        if hasattr(obj, attr):
            total += _object_nbytes(getattr(obj, attr), seen_storages)
    if total:
        return total

    numel = getattr(obj, "numel", None)
    element_size = getattr(obj, "element_size", None)
    if numel is not None and element_size is not None:
        try:
            return int(_call_or_value(numel) * _call_or_value(element_size))
        except Exception:
            return 0
    return 0


def _print_param_memory(model: torch.nn.Module) -> None:
    """Print memory consumed by model parameters, gradients, and buffers."""
    seen_storages: set[int] = set()
    param_bytes = sum(_object_nbytes(p, seen_storages) for p in model.parameters())
    grad_bytes = sum(_object_nbytes(p.grad, seen_storages)
                     for p in model.parameters() if p.grad is not None)
    buffers_attr = getattr(model, "buffers", None)
    if callable(buffers_attr):
        buffers_iter = buffers_attr()
    elif buffers_attr is not None:
        buffers_iter = buffers_attr
    else:
        buffers_iter = ()
    buf_bytes = sum(_object_nbytes(b, seen_storages) for b in buffers_iter)
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
        snapshot = _collect_snapshot()
        try:
            _print_param_memory(model)
        except Exception as e:
            print(f"  [memory_debug] Model memory section failed: {e}")
            traceback.print_exc()
        try:
            _print_snapshot_top_allocations(snapshot, top_n=20)
        except Exception as e:
            print(f"  [memory_debug] Snapshot section failed: {e}")
            traceback.print_exc()
        try:
            _print_history_peak(snapshot, top_n=12)
        except Exception as e:
            print(f"  [memory_debug] History section failed: {e}")
            traceback.print_exc()
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
