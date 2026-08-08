from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.estimate_distributed_data_prep import (
    EstimateError,
    estimate,
    load_service_times,
)


def test_load_service_times_accepts_only_completed_code_units(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    events = [
        {"event": "unit_started", "stream": "code", "unit": "a::code"},
        {
            "event": "unit_done",
            "stream": "commits",
            "unit": "a::commits",
            "stage_timings_s": {"index": 99},
        },
        {
            "event": "unit_done",
            "stream": "code",
            "unit": "a::code",
            "stage_timings_s": {"index": 10, "pack": 2.5},
        },
        {
            "event": "unit_done",
            "stream": "code",
            "unit": "b::code",
            "stage_timings_s": {"index": 20, "pack": 5},
        },
    ]
    progress.write_text(
        "".join(json.dumps(event) + "\n" for event in events), encoding="utf-8"
    )

    times, completed = load_service_times(progress)

    assert times == [12.5, 25.0]
    assert completed == 2


def test_estimate_reports_explicit_range_and_lane_count() -> None:
    result = estimate(
        [100.0, 200.0, 300.0, 400.0],
        completed_units=4,
        total_units=20,
        worker_count=4,
        lanes_per_worker=2,
        relative_worker_speed=1.0,
        fixed_hours=0,
    )

    assert result["remaining_units"] == 16
    assert result["total_lanes"] == 8
    assert result["ideal_wall_hours"] == pytest.approx(4000 / 8 / 3600)
    assert result["estimated_wall_hours"]["low"] < result["estimated_wall_hours"]["high"]
    assert "excludes unmeasured" in result["semantics"]


def test_duplicate_completion_fails_closed(tmp_path: Path) -> None:
    progress = tmp_path / "progress.jsonl"
    row = {
        "event": "unit_done",
        "stream": "code",
        "unit": "a::code",
        "stage_timings_s": {"index": 1},
    }
    progress.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n")

    with pytest.raises(EstimateError, match="duplicate completed unit"):
        load_service_times(progress)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"worker_count": 0},
        {"lanes_per_worker": 0},
        {"relative_worker_speed": 0},
        {"fixed_hours": -1},
    ],
)
def test_invalid_capacity_fails_closed(kwargs: dict[str, float]) -> None:
    inputs = {
        "completed_units": 1,
        "total_units": 2,
        "worker_count": 1,
        "lanes_per_worker": 1,
        "relative_worker_speed": 1.0,
        "fixed_hours": 0.0,
    }
    inputs.update(kwargs)
    with pytest.raises(EstimateError):
        estimate([1.0], **inputs)
