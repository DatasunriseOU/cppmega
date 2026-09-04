from cppmega.recipes.dense_mini_experiment import (
    CONTEXT_QUOTAS,
    FAMILY,
    H200_GBS_MBS,
    H200_TOTAL_PARAMS,
    TRAINING_MATRIX_KEYS,
    dp8_micro_batch,
    h200_micro_batch,
    h200_parameter_ledger,
    training_matrix_ready,
)
from scripts.nebius_h200_megatron_cpp_world_sweep import (
    dense_gqa_launch_contract,
    production_dsa_launch_contract,
)


def test_h200_parameter_ledger_matches_receipt():
    ledger = h200_parameter_ledger()

    assert ledger["family"] == FAMILY
    assert ledger["total"] == H200_TOTAL_PARAMS
    assert sum(ledger["parts"].values()) == H200_TOTAL_PARAMS
    assert ledger["parts"]["ngram_hash"] == 128_309_472
    assert ledger["parts"]["structure_core"] == 83_522


def test_dp8_ladder_is_divisible():
    for seq in (1024, 2048, 4096, 8192, 16384):
        gbs, mbs = dp8_micro_batch(seq, 8)
        assert gbs % (8 * mbs) == 0


def test_h200_ladder_and_context_quotas_cover_64k():
    assert sum(CONTEXT_QUOTAS.values()) == 1.0
    assert set(CONTEXT_QUOTAS) == set(H200_GBS_MBS)
    for seq, (gbs, mbs) in H200_GBS_MBS.items():
        assert h200_micro_batch(seq) == (gbs, mbs)
        assert gbs == mbs
        assert gbs * seq == 196608


def test_training_matrix_ready_is_fail_closed():
    empty = {key: False for key in TRAINING_MATRIX_KEYS}
    assert training_matrix_ready(empty) is False
    ready = {key: True for key in TRAINING_MATRIX_KEYS}
    assert training_matrix_ready(ready) is True
    ready["ci_stream"] = False
    assert training_matrix_ready(ready) is False


def test_dense_gqa_launch_contract_does_not_mutate_canonical_profile():
    args, spec = dense_gqa_launch_contract()

    assert "--group-query-attention" in args
    assert spec == (
        "cppmega.megatron.nam56r_noconv_spec",
        "build_cppmega_nam56r_noconv_stack_spec",
    )
    assert "--experimental-attention-variant" not in args


def test_dsa_launch_contract_stays_explicit_opt_in():
    args, spec = production_dsa_launch_contract()

    assert "--experimental-attention-variant" in args
    assert spec[0] == "cppmega.megatron.nam56r_full_spec"
