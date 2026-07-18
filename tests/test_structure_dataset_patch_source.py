from pathlib import Path

import torch

from cppmega.megatron import structure_dataset_patch


def test_structure_batch_hook_targets_megatron_core_utils_first():
    source = (
        Path(__file__).resolve().parents[1]
        / "cppmega"
        / "megatron"
        / "structure_dataset_patch.py"
    ).read_text()

    assert "import megatron.core.utils as batch_utils" in source
    assert "import megatron.training.utils as batch_utils" in source


def test_sample_document_ids_preserve_packed_boundaries_across_source_rows():
    raw_doc_ids = torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.long)
    spans = [
        {"target_start": 0, "source_start": 0, "source_end": 4},
        {"target_start": 4, "source_start": 0, "source_end": 2},
    ]

    result = structure_dataset_patch._sample_document_ids(
        raw_doc_ids,
        spans,
        target_len=8,
    )

    assert result.tolist() == [1, 1, 2, 2, 3, 3, 0, 0]
