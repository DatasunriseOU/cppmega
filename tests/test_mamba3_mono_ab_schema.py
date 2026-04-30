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
    guarded_stage2_training_ab_stub,
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


def test_micro_gemm_only_correct_candidate_gets_zero_production_credit() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "cute_micro_gemm_cheating_receipt",
                    "implementation_class": "cute_dsl_handwritten_wgmma_micro_gemm",
                    "receipt_scope": "micro_gemm_only",
                    "shape": "cute_gemm_64x64x64",
                    "projected_bwd_bwd_ms": 1.0,
                    "covered_slots": list(BWD_BWD_OUTPUT_NAMES),
                    "correctness": {"full_boundary_pass": True, "max_abs": 0.0},
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "registers_per_thread": 64,
                        "dynamic_smem_bytes": 32768,
                        "active_blocks_per_sm": 2,
                        "theoretical_occupancy": 0.5,
                        "total_ctas": 256,
                        "h200_sm_count": 132,
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert gate["production_credit"] is False
    assert gate["production_credit_ms"] == 0.0
    assert gate["credited_output_slots"] == []
    assert gate["rejection_reasons"] == ["micro_gemm_only_receipt"]


def test_local_component_speedup_gets_zero_credit_without_integrated_timing() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "wave6_cute_local_speedup_cheating_receipt",
                    "implementation_class": "cute_dsl_masked_lkq_apply",
                    "receipt_scope": "local_component_speedup",
                    "shape": "productionish",
                    "projected_bwd_bwd_ms": 0.06355,
                    "covered_slots": list(BWD_BWD_OUTPUT_NAMES),
                    "correctness": {"full_boundary_pass": True, "max_abs": 0.0},
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "timing_scope": "local_tile_chain",
                        "registers_per_thread": 64,
                        "dynamic_smem_bytes": 32768,
                        "active_blocks_per_sm": 4,
                        "theoretical_occupancy": 0.5,
                        "total_ctas": 264,
                        "h200_sm_count": 132,
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert projection["projected_bwd_bwd_ms"] == pytest.approx(0.06355)
    assert gate["production_credit"] is False
    assert gate["production_credit_ms"] == 0.0
    assert gate["credited_output_slots"] == []
    assert gate["rejection_reasons"] == ["non_integrated_timing_receipt"]
    assert gate["integrated_timing"]["status"] == "local_component_only"
    assert gate["performance_budget"]["status"] == "pass"


def test_low_live_set_cuda_timing_regression_gets_zero_production_credit() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "wave7_cuda_row_stream_low_live_set",
                    "implementation_class": "cuda_row_stream_low_live_set",
                    "receipt_scope": "bwd_bwd_component",
                    "shape": "productionish",
                    "projected_bwd_bwd_ms": 179.76535034179688,
                    "covered_slots": list(BWD_BWD_OUTPUT_NAMES),
                    "correctness": {"full_boundary_pass": True, "max_abs": 0.0},
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "timing_scope": "integrated_full_bwd_bwd",
                        "registers_per_thread": 125,
                        "dynamic_smem_bytes": 42244,
                        "active_blocks_per_sm": 2,
                        "theoretical_occupancy": 0.25,
                        "total_ctas": 32768,
                        "h200_sm_count": 132,
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert projection["projected_bwd_bwd_ms"] == pytest.approx(179.76535034179688)
    assert projection["ratio_vs_stage2_bwd_bwd"] == pytest.approx(
        179.76535034179688 / 3.70674
    )
    assert gate["resource_metadata"]["status"] == "pass"
    assert gate["cta_count_occupancy"]["status"] == "pass"
    assert gate["cta_count_occupancy"]["ctas_per_sm"] == pytest.approx(32768 / 132)
    assert gate["integrated_timing"]["status"] == "pass"
    assert gate["performance_budget"]["status"] == "fail"
    assert gate["production_credit"] is False
    assert gate["production_credit_ms"] == 0.0
    assert gate["credited_output_slots"] == []
    assert gate["rejection_reasons"] == ["performance_budget_not_met"]


def test_cute_one_chunk_fusion_is_promising_but_zero_production_credit() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "wave7_cute_fused_state_apply_consumers_one_chunk",
                    "implementation_class": (
                        "cute_dsl_fused_state_apply_consumers_one_chunk"
                    ),
                    "receipt_scope": "local_component_speedup",
                    "shape": "cute_lkq_chain_tile",
                    "status": "promising_research_only",
                    "projected_bwd_bwd_ms": 0.063419,
                    "covered_slots": [],
                    "correctness": {
                        "full_boundary_pass": False,
                        "one_chunk_fused_consumer_pass": True,
                        "max_abs": 7.8604e-07,
                        "tolerance": 1e-5,
                    },
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "timing_scope": "one_chunk_local_tile_chain",
                        "integrated_full_slot_timing": False,
                        "registers_per_thread": 64,
                        "dynamic_smem_bytes": 32768,
                        "active_blocks_per_sm": 4,
                        "theoretical_occupancy": 0.5,
                        "total_ctas": 264,
                        "h200_sm_count": 132,
                        "removed_global_outputs_for_fused_path": [
                            "LKQ",
                            "state",
                            "apply",
                            "dpsi",
                        ],
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert records[0]["status"] == "promising_research_only"
    assert records[0]["correctness"]["one_chunk_fused_consumer_pass"] is True
    assert records[0]["correctness"]["max_abs"] <= records[0]["correctness"]["tolerance"]
    assert records[0]["metadata"]["removed_global_outputs_for_fused_path"] == [
        "LKQ",
        "state",
        "apply",
        "dpsi",
    ]
    assert projection["projected_bwd_bwd_ms"] == pytest.approx(0.063419)
    assert projection["covered_slots"] == []
    assert gate["production_credit"] is False
    assert gate["production_credit_ms"] == 0.0
    assert gate["credited_output_slots"] == []
    assert "non_integrated_timing_receipt" in gate["rejection_reasons"]
    assert "missing_required_output_slots" in gate["rejection_reasons"]
    assert "full_boundary_correctness_not_reported" in gate["rejection_reasons"]
    assert gate["integrated_timing"]["status"] == "local_component_only"
    assert gate["performance_budget"]["status"] == "pass"


def test_incomplete_slot_coverage_rejects_boundary_candidate() -> None:
    covered = [name for name in BWD_BWD_OUTPUT_NAMES if name != "dda_cs"]
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "mono_missing_one_slot",
                    "implementation_class": "cuda_monolithic_bwd_bwd",
                    "receipt_scope": "bwd_bwd_component",
                    "shape": "productionish",
                    "projected_bwd_bwd_ms": 1.0,
                    "covered_slots": covered,
                    "correctness": {"full_boundary_pass": True, "max_abs": 0.0},
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "regs_per_thread": 64,
                        "static_smem_bytes": 16384,
                        "active_blocks_per_sm": 4,
                        "occupancy": 0.5,
                        "total_ctas": 264,
                        "h200_sm_count": 132,
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert gate["production_credit"] is False
    assert gate["rejection_reasons"] == ["missing_required_output_slots"]
    assert gate["missing_output_slots"] == ["dda_cs"]
    assert gate["resource_metadata"]["status"] == "pass"
    assert gate["cta_count_occupancy"]["status"] == "pass"


def test_underfilled_scan_owner_subset_gets_zero_production_credit() -> None:
    records = candidate_component_records_from_json(
        {
            "candidate_component_records": [
                {
                    "candidate_id": "wave5_cuda_scan_owner_dv_dmimov_dssda",
                    "implementation_class": "cuda_scan_owner_bh_component",
                    "shape": "productionish",
                    "projected_bwd_bwd_ms": 14.08131217956543,
                    "covered_slots": ["dv", "dmimo_v", "dssda"],
                    "correctness": {
                        "full_boundary_pass": False,
                        "subset_pass": True,
                        "max_abs": 4.76837158203125e-07,
                    },
                    "hardware_tags": ["H200"],
                    "metadata": {
                        "registers_per_thread": 190,
                        "dynamic_smem_bytes": 68612,
                        "active_blocks_per_sm": 1,
                        "theoretical_occupancy": 0.125,
                        "total_ctas": 128,
                        "h200_sm_count": 132,
                    },
                    "modal_hygiene": {
                        "status": "pass",
                        "active_same_campaign_count": 0,
                    },
                    "reference": {"stage2_bwd_bwd_ms": 3.70674},
                }
            ]
        }
    )

    projection = component_record_projection(records[0])
    gate = projection["production_gate"]

    assert projection["projected_bwd_bwd_ms"] == pytest.approx(14.08131217956543)
    assert projection["covered_slots"] == ["dv", "dmimo_v", "dssda"]
    assert gate["production_credit"] is False
    assert gate["production_credit_ms"] == 0.0
    assert gate["credited_output_slots"] == []
    assert "missing_required_output_slots" in gate["rejection_reasons"]
    assert "full_boundary_correctness_not_reported" in gate["rejection_reasons"]
    assert "cta_count_underfilled" in gate["rejection_reasons"]
    assert "performance_budget_not_met" in gate["rejection_reasons"]
    assert "dk" in gate["missing_output_slots"]
    assert "dq" in gate["missing_output_slots"]
    assert gate["resource_metadata"]["status"] == "pass"
    assert gate["cta_count_occupancy"]["status"] == "underfilled"
    assert gate["cta_count_occupancy"]["total_ctas"] == 128
    assert gate["cta_count_occupancy"]["minimum_total_ctas"] == 132
    assert gate["cta_count_occupancy"]["ctas_per_sm"] == pytest.approx(128 / 132)


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


def test_wave3_wave4_receipt_file_gates_current_research_numbers() -> None:
    receipt_path = (
        REPO_ROOT
        / "docs/status/mamba3_mono_ab_component_receipts_wave3_wave4_2026_04_30.json"
    )
    records = candidate_component_records_from_json(receipt_path.read_text(encoding="utf-8"))
    by_id = {record["candidate_id"]: record for record in records}

    wmma = component_record_projection(by_id["wave3_cuda_wmma_triangular_chunk_owner"])
    assert wmma["projected_bwd_bwd_ms"] == pytest.approx(8.467136001586914)
    assert wmma["covered_slots"] == ["dv", "dmimo_v", "dssda"]
    assert wmma["remaining_budget_ms_to_equal_stage2_bwd_bwd"] == pytest.approx(
        -4.760396001586914
    )

    cute = component_record_projection(
        by_id["wave3_cute_handwritten_wgmma_wrong_numerics"]
    )
    assert cute["projection_status"] == "missing_timing"
    assert cute["covered_slots"] == []
    assert by_id["wave3_cute_handwritten_wgmma_wrong_numerics"]["status"] == (
        "failed_correctness"
    )
    assert by_id["wave3_cute_handwritten_wgmma_wrong_numerics"]["correctness"][
        "max_abs"
    ] == pytest.approx(17.318359)

    cute_micro = component_record_projection(
        by_id["wave4_cute_handwritten_wgmma_micro_gemm_correct"]
    )
    assert cute_micro["projection_status"] == "missing_timing"
    assert cute_micro["covered_slots"] == []
    assert cute_micro["production_gate"]["production_credit"] is False
    assert cute_micro["production_gate"]["production_credit_ms"] == 0.0
    assert cute_micro["production_gate"]["credited_output_slots"] == []
    assert "micro_gemm_only_receipt" in cute_micro["production_gate"]["rejection_reasons"]
    assert "h200_hardware_tag_missing" not in cute_micro["production_gate"]["rejection_reasons"]
    assert "modal_hygiene_not_clean" not in cute_micro["production_gate"]["rejection_reasons"]
    assert by_id["wave4_cute_handwritten_wgmma_micro_gemm_correct"][
        "hardware_tags"
    ] == ["H200"]
    assert by_id["wave4_cute_handwritten_wgmma_micro_gemm_correct"][
        "modal_hygiene"
    ]["active_same_campaign_count"] == 0

    wgmma_plan = component_record_projection(by_id["wave3_wgmma_plan_budget"])
    assert wgmma_plan["projection_status"] == "missing_timing"
    assert wgmma_plan["gate_budget"]["green_full_kernel_ms"] == pytest.approx(3.35)
    assert wgmma_plan["gate_budget"]["chunk_owner_main_body_ms"] == pytest.approx(3.2)
    assert wgmma_plan["gate_budget"]["chunk_owner_dmimov_reducer_ms"] == pytest.approx(
        0.05
    )

    wave4_diag = component_record_projection(by_id["wave4_rr_diag_cuda_timestep_cta"])
    assert wave4_diag["projected_bwd_bwd_ms"] == pytest.approx(2.0560)
    assert wave4_diag["covered_slots"] == ["dk", "dq", "dgamma_diag"]

    scan_owner = component_record_projection(
        by_id["wave5_cuda_scan_owner_dv_dmimov_dssda"]
    )
    scan_gate = scan_owner["production_gate"]
    assert scan_owner["projected_bwd_bwd_ms"] == pytest.approx(14.08131217956543)
    assert scan_owner["covered_slots"] == ["dv", "dmimo_v", "dssda"]
    assert scan_gate["production_credit"] is False
    assert scan_gate["production_credit_ms"] == 0.0
    assert "dk" in scan_gate["missing_output_slots"]
    assert "dq" in scan_gate["missing_output_slots"]
    assert "missing_required_output_slots" in scan_gate["rejection_reasons"]
    assert "cta_count_underfilled" in scan_gate["rejection_reasons"]
    assert scan_gate["cta_count_occupancy"]["status"] == "underfilled"
    assert scan_gate["cta_count_occupancy"]["total_ctas"] == 128
    assert scan_gate["cta_count_occupancy"]["minimum_total_ctas"] == 132
    assert by_id["wave5_cuda_scan_owner_dv_dmimov_dssda"]["correctness"][
        "subset_pass"
    ] is True

    cute_fused = component_record_projection(
        by_id["wave6_cute_fused_masked_lkq_apply_tile_chain"]
    )
    cute_gate = cute_fused["production_gate"]
    assert cute_fused["projected_bwd_bwd_ms"] == pytest.approx(0.06355)
    assert cute_fused["covered_slots"] == []
    assert cute_gate["production_credit"] is False
    assert cute_gate["production_credit_ms"] == 0.0
    assert cute_gate["credited_output_slots"] == []
    assert "non_integrated_timing_receipt" in cute_gate["rejection_reasons"]
    assert "missing_required_output_slots" in cute_gate["rejection_reasons"]
    assert cute_gate["integrated_timing"]["status"] == "local_component_only"
    assert by_id["wave6_cute_fused_masked_lkq_apply_tile_chain"]["correctness"][
        "max_abs"
    ] == pytest.approx(7.6294e-06)
    assert by_id["wave6_cute_fused_masked_lkq_apply_tile_chain"]["correctness"][
        "tolerance"
    ] == pytest.approx(1e-5)
    assert by_id["wave6_cute_fused_masked_lkq_apply_tile_chain"]["metadata"][
        "scalar_chain_us"
    ] == pytest.approx(104.710)
    assert by_id["wave6_cute_fused_masked_lkq_apply_tile_chain"]["metadata"][
        "fused_chain_us"
    ] == pytest.approx(63.550)

    row_stream = component_record_projection(
        by_id["wave7_cuda_row_stream_low_live_set"]
    )
    row_gate = row_stream["production_gate"]
    assert row_stream["projected_bwd_bwd_ms"] == pytest.approx(179.76535034179688)
    assert row_stream["covered_slots"] == ["dv", "dmimo_v", "dssda"]
    assert row_gate["production_credit"] is False
    assert row_gate["production_credit_ms"] == 0.0
    assert "performance_budget_not_met" in row_gate["rejection_reasons"]
    assert row_gate["cta_count_occupancy"]["status"] == "pass"
    assert row_gate["resource_metadata"]["present"]["active_blocks_per_sm"] == 2
    assert by_id["wave7_cuda_row_stream_low_live_set"]["metadata"][
        "ratio_vs_tilelang_stage2_bwd_bwd"
    ] == pytest.approx(48.496886844450074)
    assert by_id["wave7_cuda_row_stream_low_live_set"]["correctness"][
        "subset_pass"
    ] is True

    cute_consumers = component_record_projection(
        by_id["wave7_cute_fused_state_apply_consumers_one_chunk"]
    )
    cute_consumers_gate = cute_consumers["production_gate"]
    assert by_id["wave7_cute_fused_state_apply_consumers_one_chunk"]["status"] == (
        "promising_research_only"
    )
    assert cute_consumers["projected_bwd_bwd_ms"] == pytest.approx(0.063419)
    assert cute_consumers["covered_slots"] == []
    assert cute_consumers_gate["production_credit"] is False
    assert cute_consumers_gate["production_credit_ms"] == 0.0
    assert "non_integrated_timing_receipt" in cute_consumers_gate["rejection_reasons"]
    assert "missing_required_output_slots" in cute_consumers_gate["rejection_reasons"]
    assert cute_consumers_gate["integrated_timing"]["status"] == "local_component_only"
    assert by_id["wave7_cute_fused_state_apply_consumers_one_chunk"]["correctness"][
        "one_chunk_fused_consumer_pass"
    ] is True
    assert by_id["wave7_cute_fused_state_apply_consumers_one_chunk"]["metadata"][
        "fused_consumer_us"
    ] == pytest.approx(63.419)
    assert by_id["wave7_cute_fused_state_apply_consumers_one_chunk"]["metadata"][
        "removed_global_outputs_for_fused_path"
    ] == ["LKQ", "state", "apply", "dpsi"]

    productionish_records = filter_candidate_component_records_for_shape(
        records,
        "productionish",
    )
    assert {record["candidate_id"] for record in productionish_records} == {
        "wave3_cuda_wmma_triangular_chunk_owner",
        "wave3_wgmma_plan_budget",
        "wave4_rr_diag_cuda_timestep_cta",
        "wave5_cuda_scan_owner_dv_dmimov_dssda",
        "wave7_cuda_row_stream_low_live_set",
    }


def test_guarded_stage2_training_ab_stub_is_opt_in_and_reversible() -> None:
    stub = guarded_stage2_training_ab_stub(run_id="wave4_train_ab", train_iters=12)

    assert stub["production_defaults_changed"] is False
    assert stub["reference"]["commit"] == MAIN_GUARDED_STAGE2_COMMIT
    assert stub["reference"]["bb_num_stages"] == 0

    commands = "\n".join(
        command
        for leg in stub["launcher_stub"].values()
        for command in leg
    )
    assert "MAMBA3_STAGE2_FORCE_NONTMA_ALLOW_FILE_MUTATION=1" in commands
    assert "CPPMEGA_MAMBA3_STAGE2_FORCE_NONTMA_ROLLBACK=1" in commands
    assert "RUN_ID=wave4_train_ab_baseline TRAIN_ITERS=12" in commands
    assert "RUN_ID=wave4_train_ab_stage2_bf1bb0 TRAIN_ITERS=12" in commands
