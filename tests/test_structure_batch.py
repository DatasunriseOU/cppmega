from cppmega.megatron.structure_batch import extract_structure_inputs, maybe_set_structure_inputs


class _Model:
    def __init__(self):
        self.seen = None

    def set_cppmega_structure_inputs(self, structure_inputs):
        self.seen = structure_inputs


def test_extract_structure_inputs_filters_only_known_keys():
    batch = {
        "tokens": 1,
        "structure_ids": 2,
        "dep_levels": 3,
        "junk": 4,
    }

    extracted = extract_structure_inputs(batch)

    assert extracted == {"structure_ids": 2, "dep_levels": 3}


def test_maybe_set_structure_inputs_pushes_non_zero_metadata_into_model():
    model = _Model()
    batch = {
        "structure_ids": [[1, 2], [3, 4]],
        "dep_levels": [[0, 1], [1, 2]],
    }

    result = maybe_set_structure_inputs(model, batch)

    assert result is not None
    assert result["structure_ids"][0][0] == 1
    assert model.seen == result
