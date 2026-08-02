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
