from types import SimpleNamespace

from cppmega.features.m2rnn import CppMegaM2RNNConfig, build_cppmega_m2rnn_config


def test_build_cppmega_m2rnn_config_maps_expected_fields():
    megatron_config = SimpleNamespace(
        hidden_size=512,
        m2rnn_k_head_dim=32,
        m2rnn_v_head_dim=8,
        m2rnn_conv_kernel=5,
        m2rnn_gradient_clipping=0.5,
        m2rnn_use_xma=False,
    )

    assert build_cppmega_m2rnn_config(megatron_config) == CppMegaM2RNNConfig(
        d_model=512,
        k_head_dim=32,
        v_head_dim=8,
        conv_kernel=5,
        gradient_clipping=0.5,
        use_xma=False,
    )
