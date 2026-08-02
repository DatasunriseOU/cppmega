"""Small fail-closed checks shared by remote gate harnesses."""

from __future__ import annotations

from typing import Any


def require_variant_rows(result: dict[str, Any]) -> None:
    """Reject a gate receipt that did not execute any measured variant."""

    variants = result.get("variants")
    if isinstance(variants, list) and variants:
        return
    blocker = result.get("blocker")
    blocker = blocker if isinstance(blocker, dict) else {}
    status = result.get("status") or blocker.get("status") or "UNKNOWN"
    reason = blocker.get("reason") or "no variant rows were produced"
    run_id = result.get("run_id") or "<unknown>"
    raise RuntimeError(
        f"gate run {run_id} produced an empty summary (zero variant rows); "
        f"status={status}; reason={reason}; refusing to report success"
    )


def require_successful_steps(returncodes: dict[str, int]) -> None:
    """Reject a harness preflight when any named subprocess failed."""

    failed = {name: code for name, code in returncodes.items() if code != 0}
    if failed:
        raise RuntimeError(f"required harness steps failed: {failed}")


def require_successful_training_variants(
    result: dict[str, Any],
    *,
    expected_variants: tuple[str, ...],
    minimum_steps: int,
) -> None:
    """Reject incomplete, failed, or non-finite training variant receipts."""

    variants = result.get("variants")
    rows = variants if isinstance(variants, list) else []
    by_name = {
        row.get("variant"): row
        for row in rows
        if isinstance(row, dict) and isinstance(row.get("variant"), str)
    }
    failures: list[str] = []
    for name in expected_variants:
        row = by_name.get(name)
        if row is None:
            failures.append(f"{name}: missing")
            continue
        run = row.get("run") if isinstance(row.get("run"), dict) else {}
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        if run.get("status") != "ok" or run.get("returncode") != 0:
            failures.append(
                f"{name}: status={run.get('status')!r} returncode={run.get('returncode')!r}"
            )
        steps = metrics.get("iterations_seen")
        if not isinstance(steps, int) or steps < minimum_steps:
            failures.append(f"{name}: steps={steps!r} < {minimum_steps}")
        if not metrics.get("lm_losses"):
            failures.append(f"{name}: no lm loss values")
        if not metrics.get("grad_norms"):
            failures.append(f"{name}: no grad norm values")
        for key in (
            "nonfinite_lm_loss_count",
            "nonfinite_mtp_loss_count",
            "nonfinite_grad_norm_count",
        ):
            count = metrics.get(key)
            if not isinstance(count, int) or count != 0:
                failures.append(f"{name}: {key}={count!r}")
        nan_iterations = metrics.get("max_nan_iterations")
        if not isinstance(nan_iterations, int) or nan_iterations != 0:
            failures.append(f"{name}: max_nan_iterations={nan_iterations!r}")
    if failures:
        raise RuntimeError("training gate receipt failed: " + "; ".join(failures))
