from cppmega.megatron.mamba3_mono_ab_schema import (
    BWD_BWD_OUTPUT_NAMES,
    MAIN_GUARDED_STAGE2_COMMIT,
    candidate_configs,
    cuda_subset_slot_results,
    memory_accounting,
    selected_shapes,
    slot_results_from_diffs,
    slot_schema,
    summarize_slot_results,
)


def test_boundary_slot_shapes_for_smoke() -> None:
    shape = selected_shapes("smoke")[0]
    slots = {slot["name"]: slot for slot in slot_schema(shape)}

    assert slots["dk"]["shape"] == [1, 1024, 4, 64]
    assert slots["dq"]["shape"] == [1, 1024, 4, 64]
    assert slots["dv"]["shape"] == [1, 256, 4, 64]
    assert slots["dmimo_v"]["shape"] == [1, 4, 4, 64]
    assert slots["dangles"]["shape"] == [1, 256, 4, 16]
    assert slots["dssda"]["shape"] == [1, 4, 16, 16, 16]

    required = {
        slot["name"]
        for slot in slots.values()
        if slot["required_for_monolithic_ab"]
    }
    assert required == set(BWD_BWD_OUTPUT_NAMES)


def test_memory_accounting_includes_boundary_outputs() -> None:
    shape = selected_shapes("productionish")[0]
    accounting = memory_accounting(shape)

    assert accounting["bwd_bwd_output_bytes"] > accounting["handoff_cache_bytes"] / 4
    assert accounting["comparison_duplicate_bwd_bwd_output_bytes"] == accounting["bwd_bwd_output_bytes"]
    assert accounting["estimated_live_floor_mib"] > accounting["bwd_bwd_output_mib"]


def test_candidate_configs_include_production_reference_and_future_slots() -> None:
    configs = candidate_configs("mono_a,mono_b")
    by_id = {config["candidate_id"]: config for config in configs}

    assert by_id["main_guarded_stage2"]["source"]["commit"] == MAIN_GUARDED_STAGE2_COMMIT
    assert by_id["main_guarded_stage2"]["config"]["bb_num_stages"] == 0
    assert by_id["cuda_covered_subset_wave9"]["role"] == "prior_component_floor"
    assert by_id["mono_a"]["role"] == "future_monolithic_candidate"
    assert by_id["mono_b"]["config"]["expected_call_boundary"] == "mamba_mimo_bwd_bwd"


def test_slot_results_from_diffs_require_all_outputs() -> None:
    diffs = {
        name: {"max_abs": 0.0, "ref_absmax": 1.0, "rel_to_ref_absmax": 0.0}
        for name in BWD_BWD_OUTPUT_NAMES
    }
    results = slot_results_from_diffs(diffs)
    summary = summarize_slot_results(results)

    assert summary["full_boundary_pass"] is True
    assert summary["full_boundary_pass_count"] == len(BWD_BWD_OUTPUT_NAMES)


def test_cuda_subset_results_remain_partial() -> None:
    correctness = {
        "wave7_combined_diag_vs_wave5_timestep_post_cuda": {
            "dgamma_diag": 0.0,
            "dk_delta": 0.0,
            "dq_delta": 0.0,
        },
        "wave7_combined_dv_vs_torch_reference": {"dv_delta": 0.0},
    }
    results = cuda_subset_slot_results(correctness, dmimov_sidecar_receipt=True)
    summary = summarize_slot_results(results)

    assert summary["full_boundary_pass"] is False
    assert set(summary["partial"]) == {"dk", "dq", "dv", "dmimo_v", "dgamma_diag"}
    assert "dd" in summary["missing"]
