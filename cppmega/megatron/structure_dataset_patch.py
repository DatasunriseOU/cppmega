"""Dynamic monkey-patching for Megatron-LM dataset structure ingress.

Dynamically overrides GPTDataset.__getitem__, Megatron's TP batch bridge, and
MambaModel/GPTModel forward passes to stream token-aligned binary MMap metadata
columns with zero memory or serialization overhead.
"""

from __future__ import annotations

import os
import json
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


_STRUCTURE_BATCH_COLS = (
    "structure_ids",
    "dep_levels",
    "ast_depth_ids",
    "sibling_index_ids",
    "node_type_ids",
)


def _pop_structure_batch(batch: Dict[str, torch.Tensor] | None) -> Dict[str, torch.Tensor] | None:
    """Remove cppmega structure tensors from a Megatron batch and stash them."""
    if batch is None:
        _set_current_structure_batch(None)
        return None
    structure_batch = {
        col: batch.pop(col)
        for col in _STRUCTURE_BATCH_COLS
        if col in batch
    }
    if structure_batch:
        _set_current_structure_batch(structure_batch)
        return structure_batch
    _set_current_structure_batch(None)
    return None


def _lazy_init_side_channels(dataset: Any) -> Dict[str, Dict[str, Any]]:
    """Load JSON sidecar and initialize numpy.memmap for all defined side-channel columns."""
    if hasattr(dataset, "_side_channels_cache") and dataset._side_channels_cache is not None:
        return dataset._side_channels_cache

    dataset._side_channels_cache = {}

    if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
        return dataset._side_channels_cache

    try:
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
            print("[cppmega-patch] WARNING: bin_path is None, cannot initialize side channels", flush=True)
            return dataset._side_channels_cache

        # Check for JSON sidecar alongside the .bin file
        prefix = os.path.splitext(str(bin_path))[0]
        json_path = prefix + ".json"
        if not os.path.exists(json_path):
            # Try with .idx.json
            json_path = prefix + ".idx.json"
            if not os.path.exists(json_path):
                return dataset._side_channels_cache

        with open(json_path, "r", encoding="utf-8") as f:
            sidecar = json.load(f)

        side_paths = sidecar.get("side_channel_paths")
        if not side_paths or not isinstance(side_paths, dict):
            return dataset._side_channels_cache

        base_dir = os.path.dirname(json_path)
        for col, entry in side_paths.items():
            rel_path = entry.get("path")
            dtype_str = entry.get("dtype", "uint16")
            if not rel_path:
                continue
            path = os.path.join(base_dir, rel_path)
            if os.path.exists(path):
                mmap = np.memmap(path, mode="r", dtype=dtype_str)
                dataset._side_channels_cache[col] = {
                    "mmap": mmap,
                    "dtype": np.dtype(dtype_str),
                }
                print(f"[cppmega-patch] Mapped side-channel {col} from {path} with dtype {dtype_str}", flush=True)
    except Exception as exc:
        print(f"[cppmega-patch] WARNING: failed to initialize side channels: {exc}", flush=True)

    return dataset._side_channels_cache


def _get_absolute_token_indices(dataset: Any, idx: int) -> np.ndarray:
    """Reconstruct absolute token-level indices inside the flat bin file for the given sequence."""
    if dataset.shuffle_index is None:
        # Lazy memmap the indexes if not loaded yet
        dataset.shuffle_index = np.load(
            dataset.path_to_shuffle_index, allow_pickle=True, mmap_mode='r'
        )
        dataset.sample_index = np.load(
            dataset.path_to_sample_index, allow_pickle=True, mmap_mode='r'
        )
        dataset.document_index = np.load(
            dataset.path_to_document_index, allow_pickle=True, mmap_mode='r'
        )

    # Perform shuffle mapping
    shuffled_idx = dataset.shuffle_index[idx]

    # Get beginning and end documents and offsets
    doc_index_beg, doc_index_beg_offset = dataset.sample_index[shuffled_idx]
    doc_index_end, doc_index_end_offset = dataset.sample_index[shuffled_idx + 1]

    token_itemsize = np.dtype(dataset.dataset.index.dtype).itemsize
    seq_ptrs = dataset.dataset.index.sequence_pointers
    seq_lens = dataset.dataset.index.sequence_lengths
    # CRITICAL: ``doc_index_beg/end`` are indices INTO ``document_index`` (the
    # shuffled per-epoch document permutation), NOT raw document ids. Megatron's
    # GPTDataset._query_document_sample_shuffle_indices fetches the real document
    # via ``self.document_index[i]`` -- mirror that exactly. Using ``i`` directly
    # reads the WRONG document's pointer/length whenever document_index is a
    # permutation (the default), yielding truncated index runs (the 1170 vs 4096
    # collate crash). No silent fallback (RULE #1): wrong docs -> wrong data.
    docmap = dataset.document_index

    if doc_index_beg == doc_index_end:
        # Sequence spans a single document
        real_doc = int(docmap[doc_index_beg])
        doc_start_token = seq_ptrs[real_doc] // token_itemsize
        start = doc_start_token + doc_index_beg_offset
        length = doc_index_end_offset - doc_index_beg_offset + dataset.config.add_extra_token_to_sequence
        return np.arange(start, start + length, dtype=np.int64)
    else:
        # Sequence spans multiple documents
        parts = []
        for i in range(doc_index_beg, doc_index_end + 1):
            real_doc = int(docmap[i])
            doc_start_token = seq_ptrs[real_doc] // token_itemsize
            if i == doc_index_beg:
                start = doc_start_token + doc_index_beg_offset
                length = seq_lens[real_doc] - doc_index_beg_offset
            elif i == doc_index_end:
                start = doc_start_token
                length = doc_index_end_offset + dataset.config.add_extra_token_to_sequence
            else:
                start = doc_start_token
                length = seq_lens[real_doc]
            parts.append(np.arange(start, start + length, dtype=np.int64))
        return np.concatenate(parts)


# --- 1. Monkey-patch GPTDataset.__getitem__ ---
try:
    from megatron.core.datasets.gpt_dataset import GPTDataset

    orig_getitem = GPTDataset.__getitem__

    def patched_getitem(self: GPTDataset, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        sample = orig_getitem(self, idx)

        if os.environ.get("CPPMEGA_STRUCTURE_ENABLED", "0") != "1":
            return sample

        # Enabled structure columns (canonical names the model embedding expects).
        structure_cols = ["structure_ids", "dep_levels", "ast_depth_ids", "sibling_index_ids", "node_type_ids"]
        # RULE #1: the converter writes side-channel keys under the parquet column
        # spelling (``token_*``); older datasets used the bare canonical name. Try
        # both so OUR reindexed dataset AND the legacy clang_semantic_4k_v10 lane
        # resolve. A genuine miss RAISES below -- never a silent zero substitution.
        _STRUCTURE_COL_ALIASES = {
            "structure_ids": ("token_structure_ids", "structure_ids"),
            "dep_levels": ("token_dep_levels", "dep_levels"),
            "ast_depth_ids": ("token_ast_depth", "ast_depth_ids", "token_ast_depth_ids"),
            "sibling_index_ids": ("token_sibling_index", "sibling_index_ids", "token_sibling_index_ids"),
            "node_type_ids": ("token_ast_node_type", "node_type_ids", "token_ast_node_type_ids"),
        }

        if idx is None:
            # Padded sequence: return zero tensors matching the tokens shape
            tokens_shape = sample["tokens"].shape
            for col in structure_cols:
                sample[col] = torch.zeros(tokens_shape, dtype=torch.long, device=sample["tokens"].device)
            return sample

        # Initialize and fetch side-channels
        side_channels = _lazy_init_side_channels(self)
        if not side_channels:
            return sample

        # RULE #1: no try/except->zeros. Resolve each canonical column from the
        # sidecar under any known alias; an unresolved column while structure is
        # enabled is a real misconfiguration and RAISES with WHERE+WHAT.
        indices = _get_absolute_token_indices(self, idx)
        for col in structure_cols:
            source = next(
                (a for a in _STRUCTURE_COL_ALIASES[col] if a in side_channels), None
            )
            if source is None:
                raise KeyError(
                    f"[cppmega-patch] structure column {col!r} missing from dataset "
                    f"side-channels (tried {_STRUCTURE_COL_ALIASES[col]}; have "
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
