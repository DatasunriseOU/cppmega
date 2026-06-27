from pathlib import Path


def test_structure_batch_hook_targets_megatron_core_utils_first():
    source = (
        Path(__file__).resolve().parents[1]
        / "cppmega"
        / "megatron"
        / "structure_dataset_patch.py"
    ).read_text()

    assert "import megatron.core.utils as batch_utils" in source
    assert "import megatron.training.utils as batch_utils" in source
