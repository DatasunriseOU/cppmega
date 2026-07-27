from __future__ import annotations

import hashlib
import json

import pytest

from scripts.ci_log_sidecars import _repo_source_binding
from scripts.ci_source_binding_projection import (
    LEGACY_PARSER_SHA256,
    SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
    SOURCE_BINDING_PROJECTION_SCHEMA,
    SourceBindingProjectionError,
    SourceBindingProjector,
    projection_record_key,
    projection_script_sha256,
    summarize_projection_records,
    target_parser_script_sha256,
)

_HEAD = "a" * 40
_PROVENANCE_SHA256 = "b" * 64
_OCCURRENCE_KEY = {
    "repo": "owner/base",
    "run_attempt": "123:1",
    "job": "456",
    "step": "section:compile:0",
    "chunk_ordinal": 7,
}
_PROVENANCE = {
    "repository": "owner/base",
    "source_repository": "contributor/base",
    "workflow": {
        "event": "pull_request",
        "head_sha": _HEAD,
    },
}


def _legacy_binding(source_input: str) -> dict[str, object] | None:
    normalized = source_input.replace("\\", "/")
    marker_index = normalized.rfind("/base/")
    if marker_index >= 0:
        source_path = normalized[marker_index + len("/base/") :]
        score = 0.8
        method = "workspace_repo_basename_suffix_v1"
    elif not normalized.startswith("/"):
        source_path = normalized.removeprefix("./")
        score = 0.95
        method = "relative_source_path_v1"
    else:
        return None
    return {
        "repository": "contributor/base",
        "head_sha": _HEAD,
        "source_path": source_path,
        "confidence": {
            "score": score,
            "level": "high",
            "source": method,
        },
    }


def _action(
    source_inputs: list[str],
    bindings: list[dict[str, object]],
    *,
    cwd: str | None = "/home/runner/work/base/base/build",
) -> dict[str, object]:
    return {
        "command_sha256": "c" * 64,
        "action_shape_sha256": "d" * 64,
        "cwd": cwd,
        "source_inputs": source_inputs,
        "source_input_count": len(source_inputs),
        "repository_source_bindings": bindings,
        "repository_source_binding_count": len(bindings),
    }


def _current_bindings(
    source_inputs: list[str],
    *,
    cwd: str | None,
    provenance: dict[str, object] = _PROVENANCE,
) -> list[dict[str, object]]:
    binding_provenance = {
        "repository": provenance["repository"],
        "source_repository": provenance["source_repository"],
        "run": provenance["workflow"],
    }
    return [
        binding
        for source in source_inputs
        if (
            binding := _repo_source_binding(
                source,
                binding_provenance,
                cwd=cwd,
            )
        )
        is not None
    ]


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
    ).hexdigest()


def test_legacy_projection_requires_exact_explicit_authorization() -> None:
    with pytest.raises(
        SourceBindingProjectionError,
        match="requires exact explicit authorization",
    ):
        SourceBindingProjector(LEGACY_PARSER_SHA256)
    with pytest.raises(
        SourceBindingProjectionError,
        match="requires exact explicit authorization",
    ):
        SourceBindingProjector(
            LEGACY_PARSER_SHA256,
            authorized_legacy_sha256="0" * 64,
        )
    with pytest.raises(
        SourceBindingProjectionError,
        match="unsupported input parser",
    ):
        SourceBindingProjector("1" * 64)


def test_legacy_projection_verifies_and_corrects_each_source_input() -> None:
    source_inputs = [
        "src/main.cpp",
        "../src/nested.cpp",
        "../../../../escape.cpp",
        "/home/runner/work/base/base/lib/absolute.cpp",
    ]
    old_bindings = [
        binding
        for source in source_inputs
        if (binding := _legacy_binding(source)) is not None
    ]
    action = _action(source_inputs, old_bindings)
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )

    result = projector.project_action(
        _OCCURRENCE_KEY,
        _PROVENANCE_SHA256,
        _PROVENANCE,
        action,
        2,
    )

    assert projector.mode == "legacy_projection"
    assert len(result.records) == len(source_inputs)
    assert [binding["source_path"] for binding in result.projected_bindings] == [
        "build/src/main.cpp",
        "src/nested.cpp",
        "lib/absolute.cpp",
    ]
    assert {(record["change_kind"], record["reason"]) for record in result.records} == {
        ("modified", "repository_and_source_path_corrected"),
        ("dropped", "unsafe_or_unresolvable_binding_dropped"),
        ("modified", "pull_request_repository_corrected"),
    }
    dropped = result.records[2]
    assert dropped["old_binding"] == old_bindings[2]
    assert dropped["projected_binding"] is None
    assert (
        dropped["source_input_sha256"]
        == hashlib.sha256(source_inputs[2].encode()).hexdigest()
    )
    assert (
        dropped["cwd_sha256"] == hashlib.sha256(str(action["cwd"]).encode()).hexdigest()
    )
    assert dropped["action_sha256"] == _canonical_sha256(action)
    assert projection_record_key(result.records[0]) == (
        "owner/base",
        "123:1",
        "456",
        "section:compile:0",
        7,
        2,
        0,
    )
    assert summarize_projection_records(result.records) == {
        "source_input_count": 4,
        "old_binding_count": 4,
        "projected_binding_count": 3,
        "unchanged_count": 0,
        "modified_count": 3,
        "added_count": 0,
        "dropped_count": 1,
    }


def test_current_parser_is_audit_only_and_deterministic() -> None:
    source_inputs = ["src/main.cpp", "/outside/checkout.cpp"]
    cwd = None
    action = _action(
        source_inputs,
        _current_bindings(source_inputs, cwd=cwd),
        cwd=cwd,
    )
    projector = SourceBindingProjector(target_parser_script_sha256())

    first = projector.project_action(
        _OCCURRENCE_KEY,
        _PROVENANCE_SHA256,
        _PROVENANCE,
        action,
        0,
    )
    second = projector.project_action(
        _OCCURRENCE_KEY,
        _PROVENANCE_SHA256,
        _PROVENANCE,
        action,
        0,
    )

    assert projector.mode == "current_audit"
    assert first == second
    assert first.projected_bindings == tuple(action["repository_source_bindings"])
    assert [record["change_kind"] for record in first.records] == [
        "unchanged",
        "unchanged",
    ]
    assert [record["reason"] for record in first.records] == [
        "current_binding_verified",
        "current_binding_verified",
    ]
    assert first.records[1]["old_binding"] is None
    assert first.records[1]["projected_binding"] is None
    assert first.records[0]["cwd_sha256"] is None


def test_projection_is_exhaustive_beyond_member_sidecar_clip_limit() -> None:
    source_inputs = [f"src/file_{index}.cpp" for index in range(12)]
    old_bindings = [
        binding
        for source in source_inputs
        if (binding := _legacy_binding(source)) is not None
    ]
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )

    result = projector.project_action(
        _OCCURRENCE_KEY,
        _PROVENANCE_SHA256,
        _PROVENANCE,
        _action(source_inputs, old_bindings),
        0,
    )

    assert len(result.records) == 12
    assert len(result.projected_bindings) == 12
    assert result.records[-1]["source_index"] == 11
    assert result.projected_bindings[-1]["source_path"] == "build/src/file_11.cpp"


@pytest.mark.parametrize(
    ("parser_sha256", "authorized"),
    [
        (LEGACY_PARSER_SHA256, LEGACY_PARSER_SHA256),
        (target_parser_script_sha256(), None),
    ],
)
def test_stored_binding_drift_fails_closed(
    parser_sha256: str,
    authorized: str | None,
) -> None:
    source_inputs = ["src/main.cpp"]
    if parser_sha256 == LEGACY_PARSER_SHA256:
        bindings = [_legacy_binding(source_inputs[0])]
    else:
        bindings = _current_bindings(source_inputs, cwd=None)
    assert bindings[0] is not None
    bindings[0] = {**bindings[0], "repository": "wrong/repository"}
    projector = SourceBindingProjector(
        parser_sha256,
        authorized_legacy_sha256=authorized,
    )

    with pytest.raises(
        SourceBindingProjectionError,
        match="stored repository source bindings disagree",
    ):
        projector.project_action(
            _OCCURRENCE_KEY,
            _PROVENANCE_SHA256,
            _PROVENANCE,
            _action(source_inputs, bindings, cwd=None),
            0,
        )


def test_truncated_action_fails_closed() -> None:
    source_inputs = ["src/one.cpp"]
    binding = _legacy_binding(source_inputs[0])
    assert binding is not None
    action = _action(source_inputs, [binding])
    action["source_input_count"] = 2
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )

    with pytest.raises(SourceBindingProjectionError, match="truncated"):
        projector.project_action(
            _OCCURRENCE_KEY,
            _PROVENANCE_SHA256,
            _PROVENANCE,
            action,
            0,
        )


@pytest.mark.parametrize(
    "missing_field",
    ["source_input_count", "repository_source_binding_count"],
)
def test_exhaustive_action_counts_are_required(missing_field: str) -> None:
    source_inputs = ["src/one.cpp"]
    binding = _legacy_binding(source_inputs[0])
    assert binding is not None
    action = _action(source_inputs, [binding])
    del action[missing_field]
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )

    with pytest.raises(SourceBindingProjectionError, match=missing_field):
        projector.project_action(
            _OCCURRENCE_KEY,
            _PROVENANCE_SHA256,
            _PROVENANCE,
            action,
            0,
        )


def test_descriptor_and_script_hashes_bind_the_implementation() -> None:
    projector = SourceBindingProjector(target_parser_script_sha256())

    assert SOURCE_BINDING_PROJECTION_SCHEMA.endswith("_v1")
    assert SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN.endswith("-v1")
    assert projection_script_sha256() == projector.implementation_sha256
    assert target_parser_script_sha256() == projector.target_parser_sha256
    assert projector.descriptor() == {
        "schema": SOURCE_BINDING_PROJECTION_SCHEMA,
        "ledger_domain": SOURCE_BINDING_PROJECTION_LEDGER_DOMAIN,
        "mode": "current_audit",
        "input_parser_sha256": target_parser_script_sha256(),
        "target_parser_sha256": target_parser_script_sha256(),
        "projection_script_sha256": projection_script_sha256(),
    }


def test_record_validator_rejects_change_semantics_drift() -> None:
    source_inputs = ["src/main.cpp"]
    old_binding = _legacy_binding(source_inputs[0])
    assert old_binding is not None
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )
    record = dict(
        projector.project_action(
            _OCCURRENCE_KEY,
            _PROVENANCE_SHA256,
            _PROVENANCE,
            _action(source_inputs, [old_binding]),
            0,
        ).records[0]
    )
    record["change_kind"] = "dropped"
    record["reason"] = "unsafe_or_unresolvable_binding_dropped"

    with pytest.raises(SourceBindingProjectionError, match="old/projected"):
        projection_record_key(record)


def test_summary_requires_contiguous_source_indexes() -> None:
    source_inputs = ["src/one.cpp", "src/two.cpp"]
    old_bindings = [
        binding
        for source in source_inputs
        if (binding := _legacy_binding(source)) is not None
    ]
    projector = SourceBindingProjector(
        LEGACY_PARSER_SHA256,
        authorized_legacy_sha256=LEGACY_PARSER_SHA256,
    )
    records = [
        dict(record)
        for record in projector.project_action(
            _OCCURRENCE_KEY,
            _PROVENANCE_SHA256,
            _PROVENANCE,
            _action(source_inputs, old_bindings),
            0,
        ).records
    ]
    records[0]["source_index"] = 1
    records[1]["source_index"] = 2

    with pytest.raises(SourceBindingProjectionError, match="start at zero"):
        summarize_projection_records(records)
