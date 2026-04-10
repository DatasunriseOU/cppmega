import torch

from cppmega.megatron.custom_embedding import CppMegaLanguageModelEmbedding


def test_normalize_structure_inputs_accepts_matching_shapes():
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    structure_inputs = {
        "structure_ids": torch.ones((2, 4), dtype=torch.int32),
        "dep_levels": torch.full((2, 4), 2, dtype=torch.int64),
    }

    normalized = CppMegaLanguageModelEmbedding._normalize_structure_inputs(input_ids, structure_inputs)

    assert normalized["structure_ids"] is not None
    assert normalized["dep_levels"] is not None
    assert normalized["structure_ids"].dtype == torch.long
    assert normalized["dep_levels"].dtype == torch.long
    assert normalized["ast_depth_ids"] is None


def test_normalize_structure_inputs_rejects_shape_drift():
    input_ids = torch.zeros((2, 4), dtype=torch.long)
    structure_inputs = {"structure_ids": torch.ones((2, 3), dtype=torch.long)}

    try:
        CppMegaLanguageModelEmbedding._normalize_structure_inputs(input_ids, structure_inputs)
    except ValueError as exc:
        assert "does not match input_ids" in str(exc)
    else:
        raise AssertionError("expected ValueError for structure shape drift")
