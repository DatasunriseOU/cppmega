import pytest

torch = pytest.importorskip("torch")

from cppmega.megatron import structure_dataset_patch as patch


def test_pop_structure_batch_removes_sidecars_and_sets_thread_local():
    batch = {
        "tokens": torch.tensor([[1, 2, 3]]),
        "labels": torch.tensor([[2, 3, 4]]),
        "structure_ids": torch.tensor([[5, 6, 7]]),
        "dep_levels": torch.tensor([[0, 1, 2]]),
    }

    structure = patch._pop_structure_batch(batch)

    assert set(batch) == {"tokens", "labels"}
    assert structure is not None
    assert torch.equal(structure["structure_ids"], torch.tensor([[5, 6, 7]]))
    assert patch._get_current_structure_batch() is structure
