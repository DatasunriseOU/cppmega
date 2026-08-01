import pytest

from cppmega.megatron.upstream_patches import (
    apply_mamba3_stage2_force_nontma_patches as applier,
)


def _patched_text() -> str:
    body_markers = [
        f"    {marker}"
        for name, marker in applier._PATCHED_MARKERS.items()
        if name
        not in (
            "bf_default",
            "bb_default",
            "bwd_tma_disabled",
            "bwd_ws_disabled",
        )
    ]
    pass_configs = [
        "@tilelang.jit(",
        "    pass_configs={",
        ("        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,"),
        ("        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,"),
        "    },",
        ")",
    ]
    return "\n".join(
        [
            *pass_configs,
            "def mamba_mimo_bwd_fwd():",
            *body_markers,
            *[
                f"    T.copy(src_{index}, dst_{index}, disable_tma=True)"
                for index in range(8)
            ],
            *pass_configs,
            "def mamba_mimo_bwd_bwd():",
            "    pass",
            "def mamba_mimo_bwd_combined(",
            "    bf_num_stages=1,",
            "    bb_num_stages=0,",
            "):",
            "    pass",
        ]
    )


def test_stage2_patch_gives_bwd_fwd_qk_single_shared_producers():
    patch = applier._patch_path().read_text()
    bwd_fwd = patch[
        patch.index("def mamba_mimo_bwd_fwd_kernel") : patch.index(
            "def mamba_mimo_bwd_bwd_kernel"
        )
    ]

    expected = (
        (
            "q",
            "+                T.copy(Q[i_b, "
            "fused_chunk_start:fused_chunk_start+fused_chunk_size, "
            "i_h_qk, :], q_frag, disable_tma=True)",
            "+                T.copy(Q[i_b, "
            "fused_chunk_start:fused_chunk_start+fused_chunk_size, "
            "i_h_qk, :], q_shared)",
            "+                T.copy(q_frag, q_shared)",
        ),
        (
            "k",
            "+                T.copy(K[i_b, "
            "fused_chunk_start:fused_chunk_start+fused_chunk_size, "
            "i_h_qk, :], k_frag, disable_tma=True)",
            "+                T.copy(K[i_b, "
            "fused_chunk_start:fused_chunk_start+fused_chunk_size, "
            "i_h_qk, :], k_shared)",
            "+                T.copy(k_frag, k_shared)",
        ),
    )
    for name, direct_fragment, unsafe_shared, biased_shared in expected:
        assert bwd_fwd.count(direct_fragment) == 1, name
        assert unsafe_shared not in bwd_fwd, name
        assert bwd_fwd.count(biased_shared) == 1, name


def test_stage2_single_write_qk_prepare_has_output_and_gradient_parity():
    torch = pytest.importorskip("torch")
    chunk_size, mimo_rank, state_size = 3, 2, 4

    def legacy_prepare(raw, bias):
        shared = raw.reshape(chunk_size * mimo_rank, state_size).clone()
        fragment = shared.clone().reshape(chunk_size, mimo_rank, state_size)
        fragment = fragment + bias.unsqueeze(0)
        return fragment.reshape(chunk_size * mimo_rank, state_size).clone()

    def single_write_prepare(raw, bias):
        fragment = raw.reshape(chunk_size, mimo_rank, state_size).clone()
        fragment = fragment + bias.unsqueeze(0)
        return fragment.reshape(chunk_size * mimo_rank, state_size).clone()

    q = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4) / 17
    k = torch.arange(24, dtype=torch.float64).reshape(3, 2, 4).flip(0) / 19
    q_bias = torch.arange(8, dtype=torch.float64).reshape(2, 4) / 23
    k_bias = torch.arange(8, dtype=torch.float64).reshape(2, 4).flip(1) / 29
    weights = torch.arange(1, 25, dtype=torch.float64).reshape(6, 4) / 31

    def evaluate(prepare):
        inputs = tuple(
            value.detach().clone().requires_grad_(True)
            for value in (q, k, q_bias, k_bias)
        )
        q_out = prepare(inputs[0], inputs[2])
        k_out = prepare(inputs[1], inputs[3])
        loss = ((q_out * k_out).tanh() * weights).sum()
        gradients = torch.autograd.grad(loss, inputs)
        return q_out, k_out, gradients

    legacy_q, legacy_k, legacy_gradients = evaluate(legacy_prepare)
    single_q, single_k, single_gradients = evaluate(single_write_prepare)

    torch.testing.assert_close(single_q, legacy_q, rtol=0, atol=0)
    torch.testing.assert_close(single_k, legacy_k, rtol=0, atol=0)
    for single, legacy in zip(single_gradients, legacy_gradients, strict=True):
        torch.testing.assert_close(single, legacy, rtol=0, atol=0)


def test_stage2_patch_preserves_fail_closed_bwd_pass_configs():
    patch = applier._patch_path().read_text()

    unsafe_patch_lines = (
        "-        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,",
        "-        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,",
        "+        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,",
        "+        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,",
    )
    assert not set(unsafe_patch_lines).intersection(patch.splitlines())


def test_stage2_partial_marker_ignores_baseline_bb_default():
    text = "def mamba_mimo_bwd_combined(bb_num_stages=0):\n    pass\n"

    assert not applier._is_patched(text)
    assert not applier._has_partial_stage2_markers(text)


def test_stage2_partial_marker_flags_structural_subset():
    text = "Q: T.Tensor([B, S * R, G, N], dtype)\nbb_num_stages=0"

    assert not applier._is_patched(text)
    assert applier._has_partial_stage2_markers(text)


def test_stage2_validator_accepts_expected_state(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    kernel.write_text(_patched_text())
    applier._validate_patched(kernel)
    assert applier._is_patched(kernel.read_text())


def test_stage2_validator_rejects_unsafe_bwd_pass_configs(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        applier._PATCHED_MARKERS["bwd_tma_disabled"],
        f"# {applier._PATCHED_MARKERS['bwd_tma_disabled']}",
        1,
    )
    kernel.write_text(text)

    with pytest.raises(RuntimeError, match="both backward kernels"):
        applier._validate_patched(kernel)


def test_stage2_validator_rejects_integer_pass_config_spoof(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    marker = applier._PATCHED_MARKERS["bwd_tma_disabled"]
    text = _patched_text().replace(marker, marker.replace("True", "1"), 1)
    kernel.write_text(text)

    with pytest.raises(RuntimeError, match="both backward kernels"):
        applier._validate_patched(kernel)


def test_stage2_validator_rejects_overlapping_bwd_fwd_q_write(tmp_path):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        applier._PATCHED_MARKERS["bf_q_biased_shared"],
        (
            "T.copy(\n"
            "        Q[i_b, "
            "fused_chunk_start:fused_chunk_start+fused_chunk_size, "
            "i_h_qk, :],\n"
            "        q_shared,\n"
            "    )\n"
            f"    {applier._PATCHED_MARKERS['bf_q_biased_shared']}"
        ),
        1,
    )
    kernel.write_text(text)

    with pytest.raises(RuntimeError, match="overlapping raw and biased shared writes"):
        applier._validate_patched(kernel)


@pytest.mark.parametrize(
    ("marker_name", "replacement"),
    [
        ("bf_q_biased_shared", ""),
        (
            "bf_q_biased_shared",
            "\n    ".join([applier._PATCHED_MARKERS["bf_q_biased_shared"]] * 2),
        ),
        ("bf_k_biased_shared", ""),
        (
            "bf_k_biased_shared",
            "\n    ".join([applier._PATCHED_MARKERS["bf_k_biased_shared"]] * 2),
        ),
    ],
)
def test_stage2_validator_rejects_missing_or_duplicate_biased_shared_write(
    tmp_path, marker_name, replacement
):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        applier._PATCHED_MARKERS[marker_name],
        replacement,
        1,
    )
    kernel.write_text(text)

    with pytest.raises(
        RuntimeError,
        match="exactly one biased fragment-to-shared write",
    ):
        applier._validate_patched(kernel)


@pytest.mark.parametrize(
    ("marker_name", "replacement"),
    [
        ("bf_q_direct_fragment", ""),
        (
            "bf_q_direct_fragment",
            "\n    ".join([applier._PATCHED_MARKERS["bf_q_direct_fragment"]] * 2),
        ),
        ("bf_k_direct_fragment", ""),
        (
            "bf_k_direct_fragment",
            "\n    ".join([applier._PATCHED_MARKERS["bf_k_direct_fragment"]] * 2),
        ),
    ],
)
def test_stage2_validator_rejects_missing_or_duplicate_direct_fragment_load(
    tmp_path, marker_name, replacement
):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        applier._PATCHED_MARKERS[marker_name],
        replacement,
        1,
    )
    kernel.write_text(text)

    with pytest.raises(
        RuntimeError,
        match="exactly one direct global-to-fragment load",
    ):
        applier._validate_patched(kernel)


@pytest.mark.parametrize(
    ("name", "expected", "unsafe"),
    [
        ("bf_num_stages", 1, 0),
        ("bb_num_stages", 0, 1),
    ],
)
def test_stage2_validator_reads_actual_stage_defaults_not_comments(
    tmp_path, name, expected, unsafe
):
    kernel = tmp_path / "mamba3_mimo_bwd.py"
    text = _patched_text().replace(
        f"{name}={expected},",
        f"{name}={unsafe},  # {name}={expected}",
        1,
    )
    kernel.write_text(text)

    with pytest.raises(RuntimeError, match="backward stage defaults must stay exact"):
        applier._validate_patched(kernel)
