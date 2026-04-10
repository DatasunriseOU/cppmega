from cppmega.megatron.nam56r_layout import load_attention_layer_numbers, load_dsa_a_layer_ranks


def test_load_attention_layer_numbers_matches_nam56r_a_positions(monkeypatch):
    monkeypatch.setenv("CPPMEGA_NEM_PATTERN", "AEMEAEMEAEMR")
    monkeypatch.setenv("CPPMEGA_LAYER_DEPTH", "52")

    values = load_attention_layer_numbers()

    assert len(values) == 13
    assert values[:4] == (1, 5, 9, 13)


def test_load_dsa_a_layer_ranks_reads_env(monkeypatch):
    monkeypatch.setenv("CPPMEGA_DSA_A_LAYER_RANKS", "8,9,10,11")

    assert load_dsa_a_layer_ranks() == (8, 9, 10, 11)


def test_selective_attention_filters_pp_layer_offset_from_mla_path(monkeypatch):
    from types import SimpleNamespace

    from cppmega.megatron import nam56r_full_spec

    init_calls = []

    mla_spec = SimpleNamespace(submodules="mla-submodules")
    dsa_wrapped_spec = SimpleNamespace(submodules="dsa-submodules")

    monkeypatch.setattr(nam56r_full_spec, "build_mla_attention_layer_spec", lambda config: mla_spec)
    monkeypatch.setattr(nam56r_full_spec, "get_dsa_module_spec_for_backend", lambda **kwargs: "dsa-self-attn-spec")
    monkeypatch.setattr(
        nam56r_full_spec,
        "build_attention_layer_spec_from_self_attention_spec",
        lambda config, self_attention_spec: dsa_wrapped_spec,
    )
    monkeypatch.setattr(nam56r_full_spec, "_get_backend_spec_provider", lambda config: "backend")

    def fake_transformer_layer_init(self, **kwargs):
        init_calls.append(kwargs)
        self.layer_number = kwargs["layer_number"]
        self.submodules_config = kwargs["submodules"]

    monkeypatch.setattr(nam56r_full_spec.TransformerLayer, "__init__", fake_transformer_layer_init)

    config = object()
    attention_layers = (1, 5, 9, 13)

    nam56r_full_spec.CppMegaSelectiveAttentionLayer(
        config=config,
        layer_number=1,
        pg_collection="pg",
        pp_layer_offset=7,
        is_mtp_layer=True,
        dsa_a_layer_ranks=(1,),
        attention_layer_numbers=attention_layers,
    )
    nam56r_full_spec.CppMegaSelectiveAttentionLayer(
        config=config,
        layer_number=5,
        pg_collection="pg",
        pp_layer_offset=7,
        is_mtp_layer=True,
        dsa_a_layer_ranks=(1,),
        attention_layer_numbers=attention_layers,
    )

    mla_kwargs = init_calls[0]
    dsa_kwargs = init_calls[1]

    assert mla_kwargs == {
        "config": config,
        "submodules": "mla-submodules",
        "layer_number": 1,
        "pg_collection": "pg",
        "add_layer_offset": False,
        "pp_layer_offset": 7,
        "is_mtp_layer": True,
    }
    assert dsa_kwargs == {
        "config": config,
        "submodules": "dsa-submodules",
        "layer_number": 5,
        "pg_collection": "pg",
        "add_layer_offset": False,
        "pp_layer_offset": 7,
        "is_mtp_layer": True,
    }


def test_selective_attention_isolates_experimental_attention_variant_per_branch(monkeypatch):
    from types import SimpleNamespace

    from cppmega.megatron import nam56r_full_spec

    mla_configs = []
    dsa_configs = []
    provider_configs = []
    init_calls = []

    def fake_build_mla_attention_layer_spec(config):
        mla_configs.append(config)
        return SimpleNamespace(submodules="mla-submodules")

    def fake_get_dsa_module_spec_for_backend(*, config, backend):
        dsa_configs.append((config, backend))
        return "dsa-self-attn-spec"

    def fake_build_attention_layer_spec_from_self_attention_spec(config, self_attention_spec):
        dsa_configs.append((config, self_attention_spec))
        return SimpleNamespace(submodules="dsa-submodules")

    def fake_get_backend_spec_provider(config):
        provider_configs.append(config)
        return "backend"

    def fake_transformer_layer_init(self, **kwargs):
        init_calls.append(kwargs)
        self.layer_number = kwargs["layer_number"]

    monkeypatch.setattr(
        nam56r_full_spec, "build_mla_attention_layer_spec", fake_build_mla_attention_layer_spec
    )
    monkeypatch.setattr(
        nam56r_full_spec, "get_dsa_module_spec_for_backend", fake_get_dsa_module_spec_for_backend
    )
    monkeypatch.setattr(
        nam56r_full_spec,
        "build_attention_layer_spec_from_self_attention_spec",
        fake_build_attention_layer_spec_from_self_attention_spec,
    )
    monkeypatch.setattr(
        nam56r_full_spec, "_get_backend_spec_provider", fake_get_backend_spec_provider
    )
    monkeypatch.setattr(nam56r_full_spec.TransformerLayer, "__init__", fake_transformer_layer_init)

    config = SimpleNamespace(experimental_attention_variant="dsa")
    attention_layers = (1, 5, 9, 13)

    nam56r_full_spec.CppMegaSelectiveAttentionLayer(
        config=config,
        layer_number=1,
        dsa_a_layer_ranks=(1,),
        attention_layer_numbers=attention_layers,
    )
    nam56r_full_spec.CppMegaSelectiveAttentionLayer(
        config=config,
        layer_number=5,
        dsa_a_layer_ranks=(1,),
        attention_layer_numbers=attention_layers,
    )

    assert config.experimental_attention_variant == "dsa"
    assert len(mla_configs) == 1
    assert mla_configs[0] is not config
    assert mla_configs[0].experimental_attention_variant is None

    assert len(provider_configs) == 1
    assert provider_configs[0] is not config
    assert provider_configs[0].experimental_attention_variant == "dsa"

    dsa_builder_config, dsa_builder_backend = dsa_configs[0]
    assert dsa_builder_config is provider_configs[0]
    assert dsa_builder_backend == "backend"

    dsa_wrap_config, dsa_wrap_spec = dsa_configs[1]
    assert dsa_wrap_config is provider_configs[0]
    assert dsa_wrap_spec == "dsa-self-attn-spec"

    assert init_calls[0]["config"] is mla_configs[0]
    assert init_calls[1]["config"] is provider_configs[0]
