#!/usr/bin/env python3
"""Estimate distributed source-map wall time from conveyor progress receipts.

The estimator deliberately reports a range.  Repository service times have a
heavy tail, so dividing the current elapsed time by a VM count produces a
misleadingly precise answer.  Only completed ``unit_done`` records with a
complete ``stage_timings_s`` mapping are accepted.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median
from typing import Iterable


class EstimateError(RuntimeError):
    """The input cannot support an evidence-bound estimate."""


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        raise EstimateError("no completed service times")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.floor(percentile * len(ordered)))
    return ordered[index]


def load_service_times(path: Path) -> tuple[list[float], int]:
    service_times: list[float] = []
    completed_units: set[str] = set()
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EstimateError(
                    f"invalid JSON at {path}:{line_number}: {exc}"
                ) from exc
            if event.get("event") != "unit_done" or event.get("stream") != "code":
                continue
            unit = event.get("unit")
            timings = event.get("stage_timings_s")
            if not isinstance(unit, str) or not unit:
                raise EstimateError(f"unit_done lacks unit at {path}:{line_number}")
            if unit in completed_units:
                raise EstimateError(f"duplicate completed unit in progress log: {unit}")
            if not isinstance(timings, dict) or not timings:
                raise EstimateError(f"unit_done lacks stage timings: {unit}")
            seconds = 0.0
            for stage, value in timings.items():
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise EstimateError(f"invalid timing {unit}/{stage}: {value!r}")
                if not math.isfinite(float(value)) or float(value) < 0:
                    raise EstimateError(f"invalid timing {unit}/{stage}: {value!r}")
                seconds += float(value)
            if seconds <= 0:
                raise EstimateError(f"non-positive service time for {unit}")
            completed_units.add(unit)
            service_times.append(seconds)
    if not service_times:
        raise EstimateError(f"no completed code units in {path}")
    return service_times, len(completed_units)


def estimate(
    service_times: Iterable[float],
    *,
    completed_units: int,
    total_units: int,
    worker_count: int,
    lanes_per_worker: int,
    relative_worker_speed: float,
    fixed_hours: float,
) -> dict[str, object]:
    samples = [float(value) for value in service_times]
    if total_units < completed_units:
        raise EstimateError("total units is smaller than completed units")
    if worker_count < 1 or lanes_per_worker < 1:
        raise EstimateError("worker and lane counts must be positive")
    if not math.isfinite(relative_worker_speed) or relative_worker_speed <= 0:
        raise EstimateError("relative worker speed must be positive")
    if not math.isfinite(fixed_hours) or fixed_hours < 0:
        raise EstimateError("fixed hours cannot be negative")

    remaining = total_units - completed_units
    lanes = worker_count * lanes_per_worker
    observed_mean = sum(samples) / len(samples)
    observed_p50 = median(samples)
    observed_p75 = _percentile(samples, 0.75)
    observed_p90 = _percentile(samples, 0.90)

    # Mean is the unbiased aggregate-work estimator.  The low/high multipliers
    # explicitly account for faster cloud CPUs at the low end and clone,
    # scheduling, reducer, and heavy-tail straggler costs at the high end.
    aggregate_seconds = remaining * observed_mean / relative_worker_speed
    ideal_seconds = aggregate_seconds / lanes
    lower_seconds = ideal_seconds * 1.08 + fixed_hours * 3600
    upper_seconds = ideal_seconds * 1.60 + fixed_hours * 3600
    return {
        "schema": "cppmega_distributed_data_prep_estimate_v1",
        "sample_count": len(samples),
        "completed_units": completed_units,
        "remaining_units": remaining,
        "worker_count": worker_count,
        "lanes_per_worker": lanes_per_worker,
        "total_lanes": lanes,
        "relative_worker_speed": relative_worker_speed,
        "service_seconds": {
            "mean": observed_mean,
            "p50": observed_p50,
            "p75": observed_p75,
            "p90": observed_p90,
        },
        "aggregate_remaining_worker_hours": aggregate_seconds / 3600,
        "ideal_wall_hours": ideal_seconds / 3600,
        "estimated_wall_hours": {
            "low": lower_seconds / 3600,
            "high": upper_seconds / 3600,
        },
        "semantics": (
            "code-map estimate from completed stage timings; excludes unmeasured "
            "commit, PR/MR, CI export, and Megatron sealing work"
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--total-units", type=int, default=501)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lanes-per-worker", type=int, default=2)
    parser.add_argument(
        "--relative-worker-speed",
        type=float,
        default=1.0,
        help="Cloud lane speed divided by the observed local lane speed.",
    )
    parser.add_argument(
        "--fixed-hours",
        type=float,
        default=2.0,
        help="Provisioning, clone warmup, reducer, and publication allowance.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    service_times, completed = load_service_times(args.progress)
    result = estimate(
        service_times,
        completed_units=completed,
        total_units=args.total_units,
        worker_count=args.workers,
        lanes_per_worker=args.lanes_per_worker,
        relative_worker_speed=args.relative_worker_speed,
        fixed_hours=args.fixed_hours,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
