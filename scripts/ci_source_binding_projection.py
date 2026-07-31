#!/usr/bin/env python3
"""Deterministic source-binding projection for frozen CI occurrences.

The frozen stream was produced by one of two supported parser implementations:

* the current parser, whose bindings are audited without modification; or
* one explicitly authorized legacy parser, whose exact historical bindings are
  verified before projecting them through the current parser semantics.

The module does not mutate occurrence provenance.  It emits canonical,
receipt-friendly ledger records and a projected binding list for a caller to
place in a derived export overlay.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import ci_log_sidecars  # noqa: E402

SOURCE_BINDING_PROJECTION_SCHEMA = "cppmega_ci_source_binding_projection_v1"
SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN = (
    "cppmega-ci-source-binding-projection-ledger-v1"
)
MAX_SOURCE_BINDING_PROJECTION_RECORD_BYTES = 4 * 1024 * 1024
LEGACY_PARSER_SHA256 = (
    "c05ff198d9b2bd817d6baa45773f08268683ef8bdc9b191c220edf1b23e1331b"
)
REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256 = (
    "e2d021137717be1011c332f63e59daf7c3d42cc293eade83d2835e3e05e14962"
)
REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON = (
    "linear giant-line path and pytest-summary scans validated on "
    "clickhouse/clickhouse run 24857659503 attempt 3"
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_OCCURRENCE_KEY_FIELDS = (
    "repo",
    "run_attempt",
    "job",
    "step",
    "chunk_ordinal",
)
_RECORD_FIELDS = frozenset(
    {
        "schema",
        "mode",
        "input_parser_sha256",
        "target_parser_sha256",
        "occurrence_key",
        "provenance_sha256",
        "action_index",
        "source_index",
        "source_input",
        "source_input_sha256",
        "cwd",
        "cwd_sha256",
        "action_sha256",
        "old_binding",
        "projected_binding",
        "change_kind",
        "reason",
    }
)


def is_reviewed_primary_equivalent_parser_transition(
    parser_lineage: Iterable[str],
    binding_upgrades: Iterable[Mapping[str, object]],
) -> bool:
    """Recognize the one reviewed parser edge that preserves primary scope."""

    current = target_parser_script_sha256()
    fields = {
        "binding_key",
        "from_sha256",
        "to_sha256",
        "reason",
        "upgraded_at",
    }
    records = tuple(binding_upgrades)
    if any(
        not isinstance(upgrade, Mapping)
        or set(upgrade) != fields
        or _HEX64_RE.fullmatch(str(upgrade.get("from_sha256"))) is None
        or _HEX64_RE.fullmatch(str(upgrade.get("to_sha256"))) is None
        or not isinstance(upgrade.get("reason"), str)
        or not upgrade["reason"]
        or re.fullmatch(
            r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z",
            str(upgrade.get("upgraded_at")),
        )
        is None
        for upgrade in records
    ):
        return False
    upgrades = [
        upgrade
        for upgrade in records
        if upgrade.get("binding_key") == "parser_script_sha256"
    ]
    if (
        tuple(parser_lineage)
        != (REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256, current)
        or len(upgrades) != 1
    ):
        return False
    upgrade = upgrades[0]
    return (
        upgrade.get("from_sha256")
        == REVIEWED_PRIMARY_EQUIVALENT_PARSER_SHA256
        and upgrade.get("to_sha256") == current
        and upgrade.get("reason")
        == REVIEWED_PRIMARY_EQUIVALENT_PARSER_UPGRADE_REASON
    )
_CHANGE_REASONS = {
    "unchanged": {
        "current_binding_verified",
        "legacy_binding_already_current",
    },
    "added": {"binding_added_by_current_semantics"},
    "dropped": {"unsafe_or_unresolvable_binding_dropped"},
    "modified": {
        "repository_and_source_path_corrected",
        "pull_request_repository_corrected",
        "runner_cwd_relative_path_normalized",
        "binding_semantics_corrected",
    },
}


class SourceBindingProjectionError(RuntimeError):
    """A parser identity, stored binding, or projection record is invalid."""


@dataclass(frozen=True)
class SourceBindingActionProjection:
    """Projection result for one stored build action."""

    records: tuple[dict[str, object], ...]
    projected_bindings: tuple[dict[str, Any], ...]
    selected_mode: str
    selected_input_parser_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _canonical_json_bytes(value: object, *, where: str) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourceBindingProjectionError(
            f"{where} is not canonical-JSON serializable: {exc}"
        ) from exc


def _sha256_json(value: object, *, where: str) -> str:
    return _sha256_bytes(_canonical_json_bytes(value, where=where))


def _script_sha256(module_file: str | None, *, where: str) -> str:
    if not isinstance(module_file, str) or not module_file:
        raise SourceBindingProjectionError(f"{where} source path is unavailable")
    path = Path(module_file)
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise SourceBindingProjectionError(
            f"cannot read {where} source at {path}: {exc}"
        ) from exc
    return _sha256_bytes(raw)


def target_parser_script_sha256() -> str:
    """Return the exact SHA-256 of the imported current parser source."""

    return _script_sha256(
        ci_log_sidecars.__file__,
        where="target parser",
    )


def projection_script_sha256() -> str:
    """Return the exact SHA-256 of this projection implementation."""

    return _script_sha256(__file__, where="projection implementation")


def _require_hex64(value: object, *, where: str) -> str:
    if not isinstance(value, str) or _HEX64_RE.fullmatch(value) is None:
        raise SourceBindingProjectionError(
            f"{where} must be one lowercase hexadecimal SHA-256"
        )
    return value


def _require_nonnegative_int(value: object, *, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SourceBindingProjectionError(f"{where} must be a non-negative integer")
    return value


def _canonical_occurrence_key(
    occurrence_key: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(occurrence_key, Mapping):
        raise SourceBindingProjectionError("occurrence_key must be a mapping")
    if set(occurrence_key) != set(_OCCURRENCE_KEY_FIELDS):
        raise SourceBindingProjectionError(
            "occurrence_key must contain exactly repo, run_attempt, job, "
            "step, and chunk_ordinal"
        )
    output: dict[str, object] = {}
    for field in _OCCURRENCE_KEY_FIELDS[:-1]:
        value = occurrence_key[field]
        if not isinstance(value, str) or not value:
            raise SourceBindingProjectionError(
                f"occurrence_key.{field} must be a non-empty string"
            )
        output[field] = value
    output["chunk_ordinal"] = _require_nonnegative_int(
        occurrence_key["chunk_ordinal"],
        where="occurrence_key.chunk_ordinal",
    )
    return output


def _occurrence_binding_provenance(
    provenance: Mapping[str, object],
) -> dict[str, object]:
    if not isinstance(provenance, Mapping):
        raise SourceBindingProjectionError("provenance must be a mapping")
    workflow = provenance.get("workflow")
    if not isinstance(workflow, Mapping):
        workflow = provenance.get("run")
    if not isinstance(workflow, Mapping):
        raise SourceBindingProjectionError("provenance.workflow must be a mapping")
    return {
        "repository": provenance.get("repository"),
        "source_repository": provenance.get("source_repository"),
        "run": dict(workflow),
    }


def _legacy_confidence(score: float, *, source: str) -> dict[str, object]:
    if score >= 1.0:
        level = "exact"
    elif score >= 0.8:
        level = "high"
    elif score > 0:
        level = "heuristic"
    else:
        level = "absent"
    return {
        "score": float(score),
        "level": level,
        "source": source,
    }


def _legacy_repo_source_binding(
    path: str,
    provenance: Mapping[str, object],
) -> dict[str, object] | None:
    """Exact local copy of the authorized legacy parser implementation."""

    repository = provenance.get("source_repository") or provenance.get("repository")
    run = provenance.get("run") or {}
    head_sha = run.get("head_sha") if isinstance(run, Mapping) else None
    if not isinstance(repository, str) or not repository:
        return None
    if not isinstance(head_sha, str) or not head_sha:
        return None
    repo_name = repository.rsplit("/", 1)[-1]
    normalized = path.replace("\\", "/")
    marker = f"/{repo_name}/"
    marker_index = normalized.rfind(marker)
    if marker_index >= 0:
        source_path = normalized[marker_index + len(marker) :]
        score = 0.8
        method = "workspace_repo_basename_suffix_v1"
    elif not normalized.startswith("/"):
        source_path = normalized.removeprefix("./")
        score = 0.95
        method = "relative_source_path_v1"
    else:
        return None
    return {
        "repository": repository,
        "head_sha": head_sha,
        "source_path": source_path,
        "confidence": _legacy_confidence(score, source=method),
    }


def _binding_list(
    value: object,
    *,
    where: str,
) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise SourceBindingProjectionError(f"{where} must be a list")
    output: list[dict[str, object]] = []
    for index, binding in enumerate(value):
        if not isinstance(binding, Mapping):
            raise SourceBindingProjectionError(f"{where}[{index}] must be a mapping")
        normalized = dict(binding)
        _canonical_json_bytes(normalized, where=f"{where}[{index}]")
        output.append(normalized)
    return output


def _source_inputs(action: Mapping[str, object]) -> list[str]:
    raw = action.get("source_inputs")
    if not isinstance(raw, list):
        raise SourceBindingProjectionError("action.source_inputs must be a list")
    output: list[str] = []
    for index, value in enumerate(raw):
        if not isinstance(value, str) or not value:
            raise SourceBindingProjectionError(
                f"action.source_inputs[{index}] must be a non-empty string"
            )
        output.append(value)
    declared = action.get("source_input_count")
    if _require_nonnegative_int(
        declared,
        where="action.source_input_count",
    ) != len(output):
        raise SourceBindingProjectionError(
            "action.source_inputs is truncated or its count is inconsistent"
        )
    return output


def _action_cwd(action: Mapping[str, object]) -> str | None:
    cwd = action.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise SourceBindingProjectionError("action.cwd must be a string or null")
    return cwd


def _change(
    old: Mapping[str, object] | None,
    projected: Mapping[str, object] | None,
    *,
    mode: str,
) -> tuple[str, str]:
    if mode == "current_audit":
        return "unchanged", "current_binding_verified"
    if old == projected:
        return "unchanged", "legacy_binding_already_current"
    if old is None:
        return "added", "binding_added_by_current_semantics"
    if projected is None:
        return "dropped", "unsafe_or_unresolvable_binding_dropped"
    repository_changed = old.get("repository") != projected.get("repository")
    path_changed = old.get("source_path") != projected.get("source_path")
    if repository_changed and path_changed:
        return "modified", "repository_and_source_path_corrected"
    if repository_changed:
        return "modified", "pull_request_repository_corrected"
    if path_changed:
        return "modified", "runner_cwd_relative_path_normalized"
    return "modified", "binding_semantics_corrected"


class SourceBindingProjector:
    """Verify stored bindings and project an authorized parser generation."""

    def __init__(
        self,
        input_parser_sha256: str,
        authorized_legacy_sha256: str | None = None,
    ) -> None:
        self.input_parser_sha256 = _require_hex64(
            input_parser_sha256,
            where="input_parser_sha256",
        )
        self.target_parser_sha256 = target_parser_script_sha256()
        self.implementation_sha256 = projection_script_sha256()
        if self.input_parser_sha256 == self.target_parser_sha256:
            if authorized_legacy_sha256 is not None:
                _require_hex64(
                    authorized_legacy_sha256,
                    where="authorized_legacy_sha256",
                )
            self.mode = "current_audit"
        elif self.input_parser_sha256 == LEGACY_PARSER_SHA256:
            if authorized_legacy_sha256 != LEGACY_PARSER_SHA256:
                raise SourceBindingProjectionError(
                    "legacy parser projection requires exact explicit authorization"
                )
            self.mode = "legacy_projection"
        else:
            raise SourceBindingProjectionError("unsupported input parser SHA-256")

    def descriptor(self) -> dict[str, str]:
        """Return the immutable fields to bind into an export receipt."""

        return {
            "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
            "ledger_domain": SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
            "mode": self.mode,
            "input_parser_sha256": self.input_parser_sha256,
            "target_parser_sha256": self.target_parser_sha256,
            "projection_script_sha256": self.implementation_sha256,
        }

    def project_action(
        self,
        occurrence_key: Mapping[str, object],
        provenance_sha256: str,
        provenance: Mapping[str, object],
        action: Mapping[str, object],
        action_index: int,
    ) -> SourceBindingActionProjection:
        """Verify and project one action, emitting one record per input."""

        key = _canonical_occurrence_key(occurrence_key)
        provenance_digest = _require_hex64(
            provenance_sha256,
            where="provenance_sha256",
        )
        if not isinstance(action, Mapping):
            raise SourceBindingProjectionError("action must be a mapping")
        normalized_action_index = _require_nonnegative_int(
            action_index,
            where="action_index",
        )
        inputs = _source_inputs(action)
        cwd = _action_cwd(action)
        stored = _binding_list(
            action.get("repository_source_bindings"),
            where="action.repository_source_bindings",
        )
        declared_binding_count = action.get("repository_source_binding_count")
        if _require_nonnegative_int(
            declared_binding_count,
            where="action.repository_source_binding_count",
        ) != len(stored):
            raise SourceBindingProjectionError(
                "action.repository_source_bindings is truncated or its count "
                "is inconsistent"
            )

        binding_provenance = _occurrence_binding_provenance(provenance)
        legacy_or_current: list[dict[str, object] | None] = []
        projected: list[dict[str, object] | None] = []
        for source_input in inputs:
            if self.mode == "legacy_projection":
                old = _legacy_repo_source_binding(
                    source_input,
                    binding_provenance,
                )
            else:
                old = ci_log_sidecars._repo_source_binding(
                    source_input,
                    binding_provenance,
                    cwd=cwd,
                )
            target = ci_log_sidecars._repo_source_binding(
                source_input,
                binding_provenance,
                cwd=cwd,
            )
            legacy_or_current.append(old)
            projected.append(target)

        expected_stored = [
            dict(binding) for binding in legacy_or_current if binding is not None
        ]
        if _canonical_json_bytes(
            stored,
            where="stored repository source bindings",
        ) != _canonical_json_bytes(
            expected_stored,
            where="expected repository source bindings",
        ):
            raise SourceBindingProjectionError(
                "stored repository source bindings disagree with exact "
                f"{self.mode} semantics"
            )

        action_digest = _sha256_json(action, where="action")
        cwd_digest = _sha256_text(cwd) if cwd is not None else None
        records: list[dict[str, object]] = []
        for source_index, (source_input, old, target) in enumerate(
            zip(inputs, legacy_or_current, projected, strict=True)
        ):
            change_kind, reason = _change(old, target, mode=self.mode)
            records.append(
                {
                    "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
                    "mode": self.mode,
                    "input_parser_sha256": self.input_parser_sha256,
                    "target_parser_sha256": self.target_parser_sha256,
                    "occurrence_key": dict(key),
                    "provenance_sha256": provenance_digest,
                    "action_index": normalized_action_index,
                    "source_index": source_index,
                    "source_input": source_input,
                    "source_input_sha256": _sha256_text(source_input),
                    "cwd": cwd,
                    "cwd_sha256": cwd_digest,
                    "action_sha256": action_digest,
                    "old_binding": None if old is None else dict(old),
                    "projected_binding": (None if target is None else dict(target)),
                    "change_kind": change_kind,
                    "reason": reason,
                }
            )
        return SourceBindingActionProjection(
            records=tuple(records),
            projected_bindings=tuple(
                dict(binding) for binding in projected if binding is not None
            ),
            selected_mode=self.mode,
            selected_input_parser_sha256=self.input_parser_sha256,
        )


class SourceBindingProjectionRouter:
    """Audit a CAS that can contain sidecars from one parser upgrade lineage.

    A fetch-state parser upgrade changes the state binding for future writes; it
    does not rewrite already committed occurrence provenance.  The router
    therefore preserves the full audited lineage while executing only semantics
    implemented here: the current sink and, when explicitly authorized, the
    exact legacy parser.  Unknown intermediate generations remain audit-only
    lineage nodes and can never instantiate an implicit projector.
    """

    SELECTION_POLICY = "stored-binding-semantics-current-first-v1"
    MIXED_MODE = "mixed_lineage_projection"

    def __init__(
        self,
        parser_lineage: Iterable[str],
        *,
        authorized_legacy_sha256: str | None = None,
    ) -> None:
        lineage = tuple(
            _require_hex64(value, where="parser_lineage entry")
            for value in parser_lineage
        )
        if not lineage:
            raise SourceBindingProjectionError("parser_lineage must not be empty")
        if len(set(lineage)) != len(lineage):
            raise SourceBindingProjectionError(
                "parser_lineage must not contain a cycle or repeated generation"
            )
        target = target_parser_script_sha256()
        if authorized_legacy_sha256 is not None:
            _require_hex64(
                authorized_legacy_sha256,
                where="authorized_legacy_sha256",
            )
        legacy_singleton = lineage == (LEGACY_PARSER_SHA256,)
        if not legacy_singleton and lineage[-1] != target:
            raise SourceBindingProjectionError(
                "parser_lineage must terminate at the current parser; only "
                "the exact authorized legacy singleton may have a legacy sink"
            )
        if (
            LEGACY_PARSER_SHA256 in lineage
            and authorized_legacy_sha256 != LEGACY_PARSER_SHA256
        ):
            raise SourceBindingProjectionError(
                "legacy parser lineage requires exact explicit authorization"
            )
        if (
            authorized_legacy_sha256 is not None
            and LEGACY_PARSER_SHA256 not in lineage
        ):
            raise SourceBindingProjectionError(
                "legacy parser authorization is outside parser_lineage"
            )

        projectors: list[SourceBindingProjector] = []
        if lineage[-1] == target:
            projectors.append(SourceBindingProjector(target))
        if LEGACY_PARSER_SHA256 in lineage:
            projectors.append(
                SourceBindingProjector(
                    LEGACY_PARSER_SHA256,
                    authorized_legacy_sha256=authorized_legacy_sha256,
                )
            )
        if not projectors:
            raise SourceBindingProjectionError(
                "parser_lineage has no executable supported source-binding "
                "semantics"
            )
        self.parser_lineage = lineage
        self.input_parser_sha256 = lineage[-1]
        self.target_parser_sha256 = target
        self.implementation_sha256 = projection_script_sha256()
        self._projectors = tuple(projectors)
        self.mode = (
            self.MIXED_MODE
            if len(self.parser_lineage) > 1
            else self._projectors[0].mode
        )

    def descriptor(self) -> dict[str, object]:
        """Return the immutable mixed-lineage projection contract."""

        return {
            "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
            "ledger_domain": SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
            "mode": self.mode,
            "input_parser_sha256": self.input_parser_sha256,
            "target_parser_sha256": self.target_parser_sha256,
            "projection_script_sha256": self.implementation_sha256,
            "parser_lineage": list(self.parser_lineage),
            "selection_policy": self.SELECTION_POLICY,
        }

    def project_action(
        self,
        occurrence_key: Mapping[str, object],
        provenance_sha256: str,
        provenance: Mapping[str, object],
        action: Mapping[str, object],
        action_index: int,
    ) -> SourceBindingActionProjection:
        """Select the exact stored semantics and return a current projection."""

        matches: list[SourceBindingActionProjection] = []
        errors: list[str] = []
        for projector in self._projectors:
            try:
                matches.append(
                    projector.project_action(
                        occurrence_key,
                        provenance_sha256,
                        provenance,
                        action,
                        action_index,
                    )
                )
            except SourceBindingProjectionError as exc:
                errors.append(f"{projector.mode}: {exc}")
        if not matches:
            raise SourceBindingProjectionError(
                "stored repository source bindings disagree with every "
                "executable supported parser semantics: "
                f"{'; '.join(errors)}"
            )
        selected = matches[0]
        for compatible in matches[1:]:
            if compatible.projected_bindings != selected.projected_bindings:
                raise SourceBindingProjectionError(
                    "authorized parser generations produce divergent current "
                    "source bindings"
                )
        return selected


def projection_record_key(
    record: Mapping[str, object],
) -> tuple[str, str, str, str, int, int, int]:
    """Validate a ledger record and return its canonical traversal key."""

    if not isinstance(record, Mapping) or set(record) != _RECORD_FIELDS:
        raise SourceBindingProjectionError(
            "projection record fields do not match the v1 schema"
        )
    if record.get("schema") != SOURCE_BINDING_PROJECTION_SCHEMA:
        raise SourceBindingProjectionError("projection record schema is unsupported")
    mode = record.get("mode")
    if mode not in {"legacy_projection", "current_audit"}:
        raise SourceBindingProjectionError("projection record mode is unsupported")
    _require_hex64(
        record.get("input_parser_sha256"),
        where="record.input_parser_sha256",
    )
    _require_hex64(
        record.get("target_parser_sha256"),
        where="record.target_parser_sha256",
    )
    _require_hex64(
        record.get("provenance_sha256"),
        where="record.provenance_sha256",
    )
    key = _canonical_occurrence_key(record.get("occurrence_key"))  # type: ignore[arg-type]
    action_index = _require_nonnegative_int(
        record.get("action_index"),
        where="record.action_index",
    )
    source_index = _require_nonnegative_int(
        record.get("source_index"),
        where="record.source_index",
    )
    source_input = record.get("source_input")
    if not isinstance(source_input, str) or not source_input:
        raise SourceBindingProjectionError(
            "record.source_input must be a non-empty string"
        )
    if record.get("source_input_sha256") != _sha256_text(source_input):
        raise SourceBindingProjectionError("record.source_input_sha256 is inconsistent")
    cwd = record.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise SourceBindingProjectionError("record.cwd must be a string or null")
    expected_cwd_digest = _sha256_text(cwd) if isinstance(cwd, str) else None
    if record.get("cwd_sha256") != expected_cwd_digest:
        raise SourceBindingProjectionError("record.cwd_sha256 is inconsistent")
    _require_hex64(
        record.get("action_sha256"),
        where="record.action_sha256",
    )
    for field in ("old_binding", "projected_binding"):
        binding = record.get(field)
        if binding is not None and not isinstance(binding, Mapping):
            raise SourceBindingProjectionError(
                f"record.{field} must be a mapping or null"
            )
        _canonical_json_bytes(binding, where=f"record.{field}")
    change_kind = record.get("change_kind")
    reason = record.get("reason")
    if (
        not isinstance(change_kind, str)
        or change_kind not in _CHANGE_REASONS
        or reason not in _CHANGE_REASONS[change_kind]
    ):
        raise SourceBindingProjectionError(
            "record change_kind/reason pair is unsupported"
        )
    old_binding = record.get("old_binding")
    projected_binding = record.get("projected_binding")
    if mode == "current_audit" and old_binding != projected_binding:
        raise SourceBindingProjectionError(
            "current audit record must be an unchanged verified binding"
        )
    expected_change = _change(
        old_binding if isinstance(old_binding, Mapping) else None,
        projected_binding if isinstance(projected_binding, Mapping) else None,
        mode=str(mode),
    )
    if (change_kind, reason) != expected_change:
        raise SourceBindingProjectionError(
            "record change_kind/reason disagrees with its old/projected bindings"
        )
    return (
        str(key["repo"]),
        str(key["run_attempt"]),
        str(key["job"]),
        str(key["step"]),
        int(key["chunk_ordinal"]),
        action_index,
        source_index,
    )


def summarize_projection_records(
    records: Iterable[Mapping[str, object]],
) -> dict[str, int]:
    """Return deterministic receipt counters after validating all records."""

    counts: Counter[str] = Counter()
    previous_key: tuple[str, str, str, str, int, int, int] | None = None
    for record in records:
        key = projection_record_key(record)
        if previous_key is not None and key <= previous_key:
            raise SourceBindingProjectionError(
                "projection records are not in strict canonical order"
            )
        if previous_key is None or key[:6] != previous_key[:6]:
            if key[6] != 0:
                raise SourceBindingProjectionError(
                    "projection source indexes must start at zero for each action"
                )
        elif key[6] != previous_key[6] + 1:
            raise SourceBindingProjectionError(
                "projection source indexes must be contiguous for each action"
            )
        previous_key = key
        change_kind = str(record["change_kind"])
        counts["source_input_count"] += 1
        counts[f"{change_kind}_count"] += 1
        if record.get("old_binding") is not None:
            counts["old_binding_count"] += 1
        if record.get("projected_binding") is not None:
            counts["projected_binding_count"] += 1
    return {
        "source_input_count": counts["source_input_count"],
        "old_binding_count": counts["old_binding_count"],
        "projected_binding_count": counts["projected_binding_count"],
        "unchanged_count": counts["unchanged_count"],
        "modified_count": counts["modified_count"],
        "added_count": counts["added_count"],
        "dropped_count": counts["dropped_count"],
    }
