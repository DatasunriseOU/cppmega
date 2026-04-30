from scripts.modal_mamba3_cuda_full_bwd_ab import _modal_hygiene_verdict


def test_modal_hygiene_fail_enforcement_fails_active_same_campaign_app() -> None:
    verdict = _modal_hygiene_verdict(
        {
            "phase": "after",
            "list_status": "ok",
            "same_campaign_active_entries": [
                {
                    "app_id": "ap-test",
                    "description": "cppmega-mamba3-test",
                    "tasks": 1,
                }
            ],
        },
        "fail",
    )

    assert verdict["status"] == "fail"
    assert verdict["enforcement"] == "fail"
    assert verdict["active_same_campaign_count"] == 1


def test_modal_hygiene_fail_enforcement_passes_clean_dry_run_list() -> None:
    verdict = _modal_hygiene_verdict(
        {
            "phase": "after",
            "list_status": "ok",
            "same_campaign_active_entries": [],
        },
        "fail",
    )

    assert verdict["status"] == "pass"


def test_modal_hygiene_fail_enforcement_fails_when_app_list_fails() -> None:
    verdict = _modal_hygiene_verdict(
        {
            "phase": "after",
            "list_status": "failed",
            "list_error": {"returncode": 1},
        },
        "fail",
    )

    assert verdict["status"] == "fail"
    assert verdict["enforcement"] == "fail"
    assert verdict["active_same_campaign_count"] == 0
    assert "could not list apps" in verdict["message"]
