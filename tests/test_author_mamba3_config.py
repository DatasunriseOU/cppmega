from types import SimpleNamespace

import pytest

from cppmega.features.mamba3 import AuthorMamba3Config, build_author_mamba3_config


def test_build_author_mamba3_config_maps_megatron_fields():
    megatron_config = SimpleNamespace(
        hidden_size=256,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=None,
    )

    assert build_author_mamba3_config(megatron_config) == AuthorMamba3Config(
        d_model=256,
        d_state=128,
        expand=2,
        headdim=64,
        ngroups=8,
    )


def test_build_author_mamba3_config_allows_optional_cppmega_overrides():
    megatron_config = SimpleNamespace(
        hidden_size=256,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=None,
        mamba_expand=4,
        cppmega_mamba3_chunk_size=96,
        cppmega_mamba3_is_mimo=True,
    )

    author_config = build_author_mamba3_config(megatron_config)
    assert author_config.expand == 4
    assert author_config.chunk_size == 96
    assert author_config.is_mimo is True


def test_build_author_mamba3_config_fails_closed_on_custom_num_heads_override():
    megatron_config = SimpleNamespace(
        hidden_size=256,
        mamba_state_dim=128,
        mamba_head_dim=64,
        mamba_num_groups=8,
        mamba_num_heads=4,
    )

    with pytest.raises(ValueError, match="mamba_num_heads override"):
        build_author_mamba3_config(megatron_config)
