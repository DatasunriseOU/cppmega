from pathlib import Path

import pytest

from cppmega.megatron.mamba3_mono_ab_schema import (
    BWD_BWD_OUTPUT_NAMES,
    MAIN_GUARDED_STAGE2_COMMIT,
    candidate_component_records_from_json,
    candidate_component_records_from_markdown,
    candidate_configs,
    component_record_projection,
    cuda_subset_slot_results,
    filter_candidate_component_records_for_shape,
    memory_accounting,
    selected_shapes,
    slot_results_from_diffs,
    slot_schema,
    summarize_slot_results,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


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


def test_component_record_json_computes_projection_budget() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "lane_a_diag_qkdv",
                    "lane": "A",
                    "shape": "productionish",
                    "components": [
                        {
                            "component_id": "diag",
                            "mean_ms": 1.0,
                            "covered_slots": ["dk", "dq", "dgamma_diag"],
                        },
                        {
                            "component_id": "qkdv",
                            "mean_ms": 0.5,
                            "covered_slots": ["dv"],
                        },
                    ],
                    "reference": {
                        "stage2_bwd_fwd_ms": 2.0,
                        "stage2_bwd_bwd_ms": 3.0,
                        "stage2_chain_ms": 5.5,
                    },
                }
            ]
        }
    )

    projection = component_record_projection(records[0])

    assert projection["projected_bwd_bwd_ms"] == 1.5
    assert projection["remaining_budget_ms_to_equal_stage2_bwd_bwd"] == 1.5
    assert projection["ratio_vs_stage2_bwd_bwd"] == 0.5
    assert projection["speedup_floor_vs_stage2_bwd_bwd"] == 2.0
    assert projection["stage2_chain_with_candidate_floor_ms"] == 3.5
    assert projection["stage2_chain_speedup_floor"] == 5.5 / 3.5
    assert projection["covered_slots"] == ["dk", "dv", "dq", "dgamma_diag"]
    assert "dmimo_v" in projection["missing_slots"]


def test_component_record_markdown_fence_and_shape_filter() -> None:
    markdown = """
Status text.

```json
{
  "mamba3_mono_ab_component_records": [
    {
      "candidate_id": "lane_b_dmimov",
      "lane": "B",
      "shape": "smoke",
      "mean_ms": 0.25,
      "covered_slots": ["dmimo_v"]
    }
  ]
}
```
"""
    records = candidate_component_records_from_markdown(
        markdown,
        source_path="docs/status/lane_b.md",
    )

    assert records[0]["candidate_id"] == "lane_b_dmimov"
    assert records[0]["source"]["doc"] == "docs/status/lane_b.md"
    assert filter_candidate_component_records_for_shape(records, "smoke") == records
    assert filter_candidate_component_records_for_shape(records, "productionish") == []


def test_candidate_configs_include_component_records_without_duplicate_future() -> None:
    records = candidate_component_records_from_json(
        [
            {
                "candidate_id": "mono_lane_c",
                "shape": "productionish",
                "mean_ms": 2.5,
                "covered_slots": ["dk"],
            }
        ]
    )
    configs = candidate_configs("mono_lane_c,mono_future", records)
    by_id = {config["candidate_id"]: config for config in configs}

    assert by_id["mono_lane_c"]["role"] == "external_component_candidate"
    assert by_id["mono_lane_c"]["component_projection"]["projected_bwd_bwd_ms"] == 2.5
    assert by_id["mono_future"]["role"] == "future_monolithic_candidate"
    assert [config["candidate_id"] for config in configs].count("mono_lane_c") == 1


def test_wave2_wave3_receipt_file_covers_reported_component_numbers() -> None:
    receipt_path = (
        REPO_ROOT
        / "docs/status/mamba3_mono_ab_component_receipts_wave2_wave3_2026_04_30.json"
    )
    records = candidate_component_records_from_json(receipt_path.read_text(encoding="utf-8"))
    by_id = {record["candidate_id"]: record for record in records}

    wmma = component_record_projection(by_id["wave2_cuda_wmma_state_lkq_d"])
    assert wmma["projected_bwd_bwd_ms"] == pytest.approx(8.919168281555176)
    assert wmma["covered_slots"] == ["dv", "dmimo_v", "dssda"]
    assert wmma["remaining_budget_ms_to_equal_stage2_bwd_bwd"] == pytest.approx(
        -5.212428281555176
    )

    triton = component_record_projection(by_id["wave2_triton_pruned_lower_bound"])
    assert triton["projected_bwd_bwd_ms"] == pytest.approx(8.79331)
    assert triton["covered_slots"] == []
    assert triton["missing_slots"] == list(BWD_BWD_OUTPUT_NAMES)

    wave3_diag = component_record_projection(by_id["wave3_rr_diag_timestep_cta"])
    assert wave3_diag["projected_bwd_bwd_ms"] == pytest.approx(2.6777)
    assert wave3_diag["covered_slots"] == ["dk", "dq", "dgamma_diag"]

    productionish_records = filter_candidate_component_records_for_shape(
        records,
        "productionish",
    )
    assert {record["candidate_id"] for record in productionish_records} == {
        "wave2_cuda_wmma_state_lkq_d",
        "wave2_triton_pruned_lower_bound",
        "wave3_rr_diag_timestep_cta",
    }
