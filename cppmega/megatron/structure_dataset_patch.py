"""Dynamic monkey-patching for Megatron-LM dataset structure ingress.

Dynamically overrides GPTDataset.__getitem__, Megatron's TP batch bridge, and
MambaModel/GPTModel forward passes to stream token-aligned binary MMap metadata
columns with zero memory or serialization overhead.
"""

from __future__ import annotations

import json
import os
import threading
from typing import Dict, Any, Optional, Tuple

import torch
import numpy as np

# Thread-local storage to safely pass the current batch's structure inputs to model forward
_local_storage = threading.local()

def _set_current_structure_batch(batch: Dict[str, torch.Tensor] | None) -> None:
    _local_storage.current_structure_batch = batch

def _get_current_structure_batch() -> Dict[str, torch.Tensor] | None:
    return getattr(_local_storage, "current_structure_batch", None)


_TOKEN_BATCH_COLS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
    "platform_ids",
    "symbol_ids",
    "call_targets",
    "type_refs",
    "def_use",
    "change_mask_pre",
    "change_mask_post",
)

_GRAPH_BATCH_COLS = (
    "graph_call_edges",
    "graph_call_edge_counts",
    "graph_type_edges",
    "graph_type_edge_counts",
    "graph_chunk_starts",
    "graph_chunk_ends",
    "graph_chunk_kinds",
    "graph_chunk_dep_levels",
    "graph_chunk_counts",
)

_TOKEN_COL_ALIASES = {
    "structure_ids": ("token_structure_ids", "structure_ids"),
    "dep_levels": ("token_dep_levels", "dep_levels"),
    "ast_depth_ids": ("token_ast_depth", "ast_depth_ids", "token_ast_depth_ids"),
    "sibling_index_ids": ("token_sibling_index", "sibling_index_ids", "token_sibling_index_ids"),
    "node_type_ids": ("token_ast_node_type", "node_type_ids", "token_ast_node_type_ids"),
    "platform_ids": ("token_platform_ids", "platform_ids"),
    "symbol_ids": ("token_symbol_ids", "symbol_ids"),
    "call_targets": ("token_call_targets", "call_targets"),
    "type_refs": ("token_type_refs", "type_refs"),
    "def_use": ("token_def_use", "def_use"),
    "change_mask_pre": ("token_change_mask_pre", "change_mask_pre"),
    "change_mask_post": ("token_change_mask_post", "change_mask_post"),
}

_GRAPH_ROUTE_COLS = (
    "token_call_edges",
    "token_type_edges",
    "token_chunk_starts",
    "token_chunk_ends",
    "token_chunk_kinds",
    "token_chunk_dep_levels",
)


def _pop_structure_batch(batch: Dict[str, torch.Tensor] | None) -> Dict[str, torch.Tensor] | None:
    """Remove cppmega sidecar tensors from a Megatron batch and stash them."""
    if batch is None:
        _set_current_structure_batch(None)
        return None
    structure_batch = {
        col: batch.pop(col)
        for col in _TOKEN_BATCH_COLS + _GRAPH_BATCH_COLS
        if col in batch
    }
    if structure_batch:
        _set_current_structure_batch(structure_batch)
        return structure_batch
    _set_current_structure_batch(None)
    return None


def _sidecar_json_path(dataset: Any) -> str:
    bin_path = getattr(dataset.dataset, "bin_path", None)
    if bin_path is None:
        path_prefix = getattr(dataset.dataset, "path_prefix", None)
        if path_prefix is not None:
            bin_path = f"{path_prefix}.bin"
        else:
            bin_reader = getattr(dataset.dataset, "bin_reader", None)
            if bin_reader is not None:
                bin_path = getattr(bin_reader, "_bin_path", None)

    if bin_path is None:
        raise RuntimeError("[cppmega-patch] bin_path is None, cannot initialize side channels")

    prefix = os.path.splitext(str(bin_path))[0]
    json_path = prefix + ".json"
    if not os.path.exists(json_path):
        json_path = prefix + ".idx.json"
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"[cppmega-patch] sidecar manifest not found next to {bin_path!r}; "
            f"tried {prefix + '.json'!r} and {prefix + '.idx.json'!r}"
        )
    return json_path


def _load_sidecar_manifest(dataset: Any) -> tuple[str, dict[str, Any]]:
    if hasattr(dataset, "_cppmega_sidecar_manifest") and dataset._cppmega_sidecar_manifest is not None:
        return dataset._cppmega_sidecar_manifest
    json_path = _sidecar_json_path(dataset)
    with open(json_path, "r", encoding="utf-8") as f:
        sidecar = json.load(f)
    dataset._cppmega_sidecar_manifest = (json_path, sidecar)
    return json_path, sidecar


def _lazy_init_side_channels(dataset: Any) -> Dict[str, Dict[str, Any]]:
    """Load JSON sidecar and initialize numpy.memmap for all defined side-channel columns."""
    if hasattr(dataset, "_side_channels_cache") and dataset._side_channels_cache is not None:
        return dataset._side_channels_cache

    dataset._side_channels_cache = {}

    if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
        return dataset._side_channels_cache

    json_path, sidecar = _load_sidecar_manifest(dataset)

    side_paths = sidecar.get("side_channel_paths")
    if not side_paths or not isinstance(side_paths, dict):
        raise KeyError(
            f"[cppmega-patch] side_channel_paths missing in {json_path!r} while "
            "CPPMEGA_STRUCTURE_ENABLED=1"
        )

    base_dir = os.path.dirname(json_path)
    for col, entry in side_paths.items():
        rel_path = entry.get("path")
        dtype_str = entry.get("dtype", "uint16")
        if not rel_path:
            raise ValueError(f"[cppmega-patch] side-channel {col!r} has no path in {json_path!r}")
        path = os.path.join(base_dir, rel_path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"[cppmega-patch] side-channel file for {col!r} not found: {path}"
            )
        mmap = np.memmap(path, mode="r", dtype=dtype_str)
        dataset._side_channels_cache[col] = {
            "mmap": mmap,
            "dtype": np.dtype(dtype_str),
        }
        print(f"[cppmega-patch] Mapped side-channel {col} from {path} with dtype {dtype_str}", flush=True)

    return dataset._side_channels_cache


def _lazy_init_graph_sidecars(dataset: Any) -> Dict[str, Dict[str, Any]]:
    if hasattr(dataset, "_graph_sidecars_cache") and dataset._graph_sidecars_cache is not None:
        return dataset._graph_sidecars_cache

    dataset._graph_sidecars_cache = {}
    if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0") != "1":
        return dataset._graph_sidecars_cache

    json_path, sidecar = _load_sidecar_manifest(dataset)
    if sidecar.get("graph_sidecar_schema") != "cppmega_graph_routes_v1":
        raise ValueError(
            f"[cppmega-patch] graph_sidecar_schema must be cppmega_graph_routes_v1 in {json_path!r}; "
            f"got {sidecar.get('graph_sidecar_schema')!r}"
        )
    graph_paths = sidecar.get("graph_sidecar_paths")
    if not graph_paths or not isinstance(graph_paths, dict):
        raise KeyError(
            f"[cppmega-patch] graph_sidecar_paths missing in {json_path!r} while "
            "CPPMEGA_GRAPH_ROUTES_ENABLED=1"
        )

    missing = sorted(set(_GRAPH_ROUTE_COLS) - set(graph_paths))
    if missing:
        raise KeyError(f"[cppmega-patch] graph sidecars missing required columns: {missing}")

    document_count = int(sidecar.get("document_count", len(dataset.dataset.index.sequence_lengths)))
    base_dir = os.path.dirname(json_path)
    for col, entry in graph_paths.items():
        offsets_path = os.path.join(base_dir, entry["offsets_path"])
        data_path = os.path.join(base_dir, entry["data_path"])
        if not os.path.exists(offsets_path):
            raise FileNotFoundError(f"[cppmega-patch] graph offsets file for {col!r} not found: {offsets_path}")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"[cppmega-patch] graph data file for {col!r} not found: {data_path}")

        offset_dtype = np.dtype(entry.get("offset_dtype", "int64"))
        dtype = np.dtype(entry.get("dtype", "int32"))
        offsets = np.memmap(offsets_path, mode="r", dtype=offset_dtype, shape=(document_count + 1,))
        if int(offsets[0]) != 0:
            raise ValueError(f"[cppmega-patch] graph offsets for {col!r} must start at 0")
        if np.any(np.diff(offsets) < 0):
            raise ValueError(f"[cppmega-patch] graph offsets for {col!r} are not monotonic")
        item_count = int(entry.get("item_count", int(offsets[-1])))
        if int(offsets[-1]) != item_count:
            raise ValueError(
                f"[cppmega-patch] graph offsets for {col!r} end at {int(offsets[-1])}, "
                f"manifest item_count={item_count}"
            )
        shape_tail = tuple(int(x) for x in entry.get("shape_tail", []))
        data_shape = (item_count,) + shape_tail
        data = np.memmap(data_path, mode="r", dtype=dtype, shape=data_shape)
        dataset._graph_sidecars_cache[col] = {
            "offsets": offsets,
            "data": data,
            "dtype": dtype,
            "shape_tail": shape_tail,
            "kind": entry.get("kind"),
        }
        print(
            f"[cppmega-patch] Mapped graph sidecar {col} from {data_path} "
            f"with dtype {dtype} shape {data_shape}",
            flush=True,
        )

    return dataset._graph_sidecars_cache


def _ensure_gpt_indexes(dataset: Any) -> None:
    if dataset.shuffle_index is None:
        dataset.shuffle_index = np.load(
            dataset.path_to_shuffle_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.sample_index = np.load(
            dataset.path_to_sample_index, allow_pickle=True, mmap_mode="r"
        )
        dataset.document_index = np.load(
            dataset.path_to_document_index, allow_pickle=True, mmap_mode="r"
        )


def _get_sample_token_spans(dataset: Any, idx: int) -> list[dict[str, int]]:
    """Return spans mapping a Megatron sample window back to source documents."""
    _ensure_gpt_indexes(dataset)

    shuffled_idx = int(dataset.shuffle_index[idx])
    doc_index_beg, doc_index_beg_offset = dataset.sample_index[shuffled_idx]
    doc_index_end, doc_index_end_offset = dataset.sample_index[shuffled_idx + 1]
    doc_index_beg = int(doc_index_beg)
    doc_index_end = int(doc_index_end)
    doc_index_beg_offset = int(doc_index_beg_offset)
    doc_index_end_offset = int(doc_index_end_offset)

    token_itemsize = np.dtype(dataset.dataset.index.dtype).itemsize
    seq_ptrs = dataset.dataset.index.sequence_pointers
    seq_lens = dataset.dataset.index.sequence_lengths
    docmap = dataset.document_index
    add_extra = int(dataset.config.add_extra_token_to_sequence)

    spans: list[dict[str, int]] = []
    target_start = 0
    for i in range(doc_index_beg, doc_index_end + 1):
        real_doc = int(docmap[i])
        doc_start_token = int(seq_ptrs[real_doc] // token_itemsize)
        if i == doc_index_beg:
            src_start = doc_index_beg_offset
            length = int(seq_lens[real_doc]) - src_start
        elif i == doc_index_end:
            src_start = 0
            length = doc_index_end_offset + add_extra
        else:
            src_start = 0
            length = int(seq_lens[real_doc])
        if doc_index_beg == doc_index_end:
            src_start = doc_index_beg_offset
            length = doc_index_end_offset - doc_index_beg_offset + add_extra
        if length <= 0:
            continue
        spans.append(
            {
                "real_doc": real_doc,
                "doc_start_token": doc_start_token,
                "source_start": src_start,
                "source_end": src_start + length,
                "target_start": target_start,
            }
        )
        target_start += length
    return spans


def _get_absolute_token_indices(dataset: Any, idx: int) -> np.ndarray:
    """Reconstruct absolute token-level indices inside the flat bin file for the given sequence."""
    parts = []
    for span in _get_sample_token_spans(dataset, idx):
        start = span["doc_start_token"] + span["source_start"]
        length = span["source_end"] - span["source_start"]
        parts.append(np.arange(start, start + length, dtype=np.int64))
    return np.concatenate(parts) if parts else np.empty((0,), dtype=np.int64)


def _slice_graph_doc(cache_entry: dict[str, Any], real_doc: int) -> np.ndarray:
    offsets = cache_entry["offsets"]
    start = int(offsets[real_doc])
    end = int(offsets[real_doc + 1])
    return np.asarray(cache_entry["data"][start:end])


def _cap_2d(values: list[tuple[int, int]], *, max_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((max_rows, 2), -1, dtype=torch.long)
    count = min(len(values), max_rows)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.long)
    return out, torch.tensor(count, dtype=torch.long)


def _cap_1d(values: list[int], *, max_rows: int, pad: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.full((max_rows,), pad, dtype=torch.long)
    count = min(len(values), max_rows)
    if count:
        out[:count] = torch.tensor(values[:count], dtype=torch.long)
    return out, torch.tensor(count, dtype=torch.long)


def _build_graph_route_tensors(
    graph_sidecars: Dict[str, Dict[str, Any]],
    spans: list[dict[str, int]],
    *,
    target_len: int,
    max_edges: int,
    max_chunks: int,
) -> Dict[str, torch.Tensor]:
    call_edges: list[tuple[int, int]] = []
    type_edges: list[tuple[int, int]] = []
    chunk_starts: list[int] = []
    chunk_ends: list[int] = []
    chunk_kinds: list[int] = []
    chunk_dep_levels: list[int] = []

    for span in spans:
        real_doc = span["real_doc"]
        source_start = span["source_start"]
        source_end = span["source_end"]
        target_start = span["target_start"]

        for source_name, sink in (
            ("token_call_edges", call_edges),
            ("token_type_edges", type_edges),
        ):
            rows = _slice_graph_doc(graph_sidecars[source_name], real_doc)
            for src, dst in rows:
                src_i = int(src)
                dst_i = int(dst)
                if source_start <= src_i < source_end and source_start <= dst_i < source_end:
                    adj_src = target_start + src_i - source_start
                    adj_dst = target_start + dst_i - source_start
                    if 0 <= adj_src < target_len and 0 <= adj_dst < target_len:
                        sink.append((adj_src, adj_dst))

        starts = _slice_graph_doc(graph_sidecars["token_chunk_starts"], real_doc)
        ends = _slice_graph_doc(graph_sidecars["token_chunk_ends"], real_doc)
        kinds = _slice_graph_doc(graph_sidecars["token_chunk_kinds"], real_doc)
        dep_levels = _slice_graph_doc(graph_sidecars["token_chunk_dep_levels"], real_doc)
        if not (len(starts) == len(ends) == len(kinds) == len(dep_levels)):
            raise ValueError(
                f"[cppmega-patch] chunk graph sidecar lengths disagree for document {real_doc}: "
                f"starts={len(starts)} ends={len(ends)} kinds={len(kinds)} dep_levels={len(dep_levels)}"
            )
        for start, end, kind, dep_level in zip(starts, ends, kinds, dep_levels, strict=True):
            start_i = int(start)
            end_i = int(end)
            overlap_start = max(start_i, source_start)
            overlap_end = min(end_i, source_end)
            if overlap_start >= overlap_end:
                continue
            adj_start = target_start + overlap_start - source_start
            adj_end = target_start + overlap_end - source_start
            if adj_start >= target_len:
                continue
            adj_end = min(adj_end, target_len)
            if adj_start < adj_end:
                chunk_starts.append(adj_start)
                chunk_ends.append(adj_end)
                chunk_kinds.append(int(kind))
                chunk_dep_levels.append(int(dep_level))

    graph_call_edges, graph_call_edge_counts = _cap_2d(call_edges, max_rows=max_edges)
    graph_type_edges, graph_type_edge_counts = _cap_2d(type_edges, max_rows=max_edges)
    graph_chunk_starts, graph_chunk_counts = _cap_1d(chunk_starts, max_rows=max_chunks)
    graph_chunk_ends, _ = _cap_1d(chunk_ends, max_rows=max_chunks)
    graph_chunk_kinds, _ = _cap_1d(chunk_kinds, max_rows=max_chunks)
    graph_chunk_dep_levels, _ = _cap_1d(chunk_dep_levels, max_rows=max_chunks)

    return {
        "graph_call_edges": graph_call_edges,
        "graph_call_edge_counts": graph_call_edge_counts,
        "graph_type_edges": graph_type_edges,
        "graph_type_edge_counts": graph_type_edge_counts,
        "graph_chunk_starts": graph_chunk_starts,
        "graph_chunk_ends": graph_chunk_ends,
        "graph_chunk_kinds": graph_chunk_kinds,
        "graph_chunk_dep_levels": graph_chunk_dep_levels,
        "graph_chunk_counts": graph_chunk_counts,
    }


# --- 1. Monkey-patch GPTDataset.__getitem__ ---
try:
    from megatron.core.datasets.gpt_dataset import GPTDataset

    orig_getitem = GPTDataset.__getitem__

    def patched_getitem(self: GPTDataset, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        sample = orig_getitem(self, idx)

        if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
            return sample

        if idx is None:
            # Padded sequence: return zero tensors matching the tokens shape
            tokens_shape = sample["tokens"].shape
            for col in _TOKEN_BATCH_COLS:
                sample[col] = torch.zeros(tokens_shape, dtype=torch.long, device=sample["tokens"].device)
            if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0") == "1":
                max_edges = int(os.environ.get("CPPMEGA_GRAPH_MAX_EDGES", "256"))
                max_chunks = int(os.environ.get("CPPMEGA_GRAPH_MAX_CHUNKS", "256"))
                graph = _build_graph_route_tensors(
                    {
                        "token_call_edges": {"offsets": np.array([0, 0]), "data": np.empty((0, 2), dtype=np.int32)},
                        "token_type_edges": {"offsets": np.array([0, 0]), "data": np.empty((0, 2), dtype=np.int32)},
                        "token_chunk_starts": {"offsets": np.array([0, 0]), "data": np.empty((0,), dtype=np.uint32)},
                        "token_chunk_ends": {"offsets": np.array([0, 0]), "data": np.empty((0,), dtype=np.uint32)},
                        "token_chunk_kinds": {"offsets": np.array([0, 0]), "data": np.empty((0,), dtype=np.uint16)},
                        "token_chunk_dep_levels": {"offsets": np.array([0, 0]), "data": np.empty((0,), dtype=np.uint16)},
                    },
                    [],
                    target_len=int(tokens_shape[-1]),
                    max_edges=max_edges,
                    max_chunks=max_chunks,
                )
                sample.update(graph)
            return sample

        # Initialize and fetch side-channels
        side_channels = _lazy_init_side_channels(self)
        if not side_channels:
            raise RuntimeError(
                "[cppmega-patch] no side channels loaded while CPPMEGA_STRUCTURE_ENABLED=1"
            )

        # RULE #1: no try/except->zeros. Resolve each canonical column from the
        # sidecar under any known alias; an unresolved column while structure is
        # enabled is a real misconfiguration and RAISES with WHERE+WHAT.
        indices = _get_absolute_token_indices(self, idx)
        for col in _TOKEN_BATCH_COLS:
            source = next(
                (a for a in _TOKEN_COL_ALIASES[col] if a in side_channels), None
            )
            if source is None:
                raise KeyError(
                    f"[cppmega-patch] token sidecar column {col!r} missing from dataset "
                    f"side-channels (tried {_TOKEN_COL_ALIASES[col]}; have "
                    f"{sorted(side_channels)}) while CPPMEGA_STRUCTURE_ENABLED=1"
                )
            entry = side_channels[source]
            vals = entry["mmap"][indices]
            tensor = torch.from_numpy(vals).long()
            if self.config.add_extra_token_to_sequence:
                tensor = tensor[:-1]
            tensor = tensor.contiguous()
            # Align to the (possibly pad-extended) token length. Megatron pads a
            # short trailing sample's tokens up to sequence_length; mirror that by
            # zero-padding the structure tail -- those are genuine pad positions
            # (loss-masked), so zeros are correct, NOT a silent data fallback.
            # RULE #1: a structure run LONGER than the token window means the index
            # reconstruction is wrong -> RAISE rather than silently truncate.
            target_len = int(sample["tokens"].shape[-1])
            if tensor.shape[0] > target_len:
                raise ValueError(
                    f"[cppmega-patch] structure col {col!r} len {tensor.shape[0]} > "
                    f"token len {target_len} (idx {idx}); index reconstruction bug"
                )
            if tensor.shape[0] < target_len:
                pad = torch.zeros(target_len - tensor.shape[0], dtype=tensor.dtype)
                tensor = torch.cat([tensor, pad], dim=0)
            sample[col] = tensor.contiguous()
            if idx == 0:
                print(
                    f"[cppmega-patch] Mapped side-channel {source} -> {col}",
                    flush=True,
                )

        if os.environ.get("CPPMEGA_GRAPH_ROUTES_ENABLED", "0") == "1":
            graph_sidecars = _lazy_init_graph_sidecars(self)
            if not graph_sidecars:
                raise RuntimeError(
                    "[cppmega-patch] no graph sidecars loaded while CPPMEGA_GRAPH_ROUTES_ENABLED=1"
                )
            max_edges = int(os.environ.get("CPPMEGA_GRAPH_MAX_EDGES", "256"))
            max_chunks = int(os.environ.get("CPPMEGA_GRAPH_MAX_CHUNKS", "256"))
            graph = _build_graph_route_tensors(
                graph_sidecars,
                _get_sample_token_spans(self, idx),
                target_len=int(sample["tokens"].shape[-1]),
                max_edges=max_edges,
                max_chunks=max_chunks,
            )
            sample.update(graph)

        return sample

    GPTDataset.__getitem__ = patched_getitem
    print("[cppmega-patch] Successfully patched GPTDataset.__getitem__", flush=True)
except Exception as e:
    print(f"[cppmega-patch] WARNING: failed to patch GPTDataset.__getitem__: {e}", flush=True)


# --- 2. Monkey-patch get_batch_on_this_tp_rank ---
try:
    try:
        # Megatron core_v0.18.0 moved this helper here; pretrain_gpt.py imports
        # from this module directly, so this patch must land before runpy enters
        # upstream pretrain_mamba/pretrain_hybrid.
        import megatron.core.utils as batch_utils  # type: ignore[import-not-found]
    except ImportError:
        # Older cppmega H200 trees used the training.utils location.
        import megatron.training.utils as batch_utils  # type: ignore[import-not-found]

    orig_get_batch_on_this_tp_rank = batch_utils.get_batch_on_this_tp_rank

    def patched_get_batch_on_this_tp_rank(*args, **kwargs) -> Dict[str, torch.Tensor] | None:
        batch = orig_get_batch_on_this_tp_rank(*args, **kwargs)
        _pop_structure_batch(batch)
        return batch

    batch_utils.get_batch_on_this_tp_rank = patched_get_batch_on_this_tp_rank
    print(
        f"[cppmega-patch] Successfully patched {batch_utils.__name__}.get_batch_on_this_tp_rank",
        flush=True,
    )
except Exception as e:
    print(f"[cppmega-patch] WARNING: failed to patch get_batch_on_this_tp_rank: {e}", flush=True)


# --- 3. Monkey-patch MambaModel / GPTModel forward passes ---
try:
    from megatron.core.models.mamba import MambaModel

    orig_mamba_forward = MambaModel.forward

    def patched_mamba_forward(self: MambaModel, *args, **kwargs) -> Any:
        structure_batch = _get_current_structure_batch()
        if structure_batch:
            from cppmega.megatron.structure_batch import maybe_set_structure_inputs
            maybe_set_structure_inputs(self, structure_batch)
        return orig_mamba_forward(self, *args, **kwargs)

    MambaModel.forward = patched_mamba_forward
    print("[cppmega-patch] Successfully patched MambaModel.forward", flush=True)
except ImportError:
    pass
except Exception as e:
    print(f"[cppmega-patch] WARNING: failed to patch MambaModel.forward: {e}", flush=True)


try:
    from megatron.core.models.gpt import GPTModel

    orig_gpt_forward = GPTModel.forward

    def patched_gpt_forward(self: GPTModel, *args, **kwargs) -> Any:
        structure_batch = _get_current_structure_batch()
        if structure_batch:
            from cppmega.megatron.structure_batch import maybe_set_structure_inputs
            maybe_set_structure_inputs(self, structure_batch)
        return orig_gpt_forward(self, *args, **kwargs)

    GPTModel.forward = patched_gpt_forward
    print("[cppmega-patch] Successfully patched GPTModel.forward", flush=True)
except ImportError:
    pass
except Exception as e:
    print(f"[cppmega-patch] WARNING: failed to patch GPTModel.forward: {e}", flush=True)
