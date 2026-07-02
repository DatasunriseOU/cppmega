from pathlib import Path


def test_mamba_builder_tolerates_current_megatron_arg_namespace():
    source = Path("cppmega/megatron/mamba_builder.py").read_text()

    assert 'getattr(args, "use_legacy_models", False)' in source
    assert "args.use_legacy_models" not in source
    assert 'getattr(args, "padded_vocab_size", None) or getattr(args, "vocab_size")' in source


def test_cppmega_mamba_model_does_not_forward_spec_decode_to_hybrid_model():
    source = Path("cppmega/megatron/custom_mamba_model.py").read_text()

    assert "is_spec_decode=None" in source
    assert "is_spec_decode=is_spec_decode" not in source
