from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.distributed_data_prep._common import (
    ContractError,
    atomic_write_json,
    canonical_sha256,
)
from scripts.distributed_data_prep.adjudicate_source_pilot import (
    AUTHORITATIVE_ARCHIVE_GENERATION,
    AUTHORITATIVE_ARTIFACT_SET_SHA256,
    AUTHORITATIVE_RECEIPT_GENERATION,
    DEFAULT_ADJUDICATION_PATH,
    EXPECTED_ADJUDICATION_SHA256,
    SOURCE_PILOT_SELECTION_SCHEMA,
    SUPERSEDED_ARTIFACT_SET_SHA256,
    SupersededPublicationError,
    load_source_pilot_adjudication,
    load_source_pilot_selection,
    select_source_pilot_publication,
    write_selection_receipt,
)

_FIXTURES = Path(__file__).parent / "fixtures"
_CANONICAL_RECEIPT = (
    _FIXTURES / "source_pilot_20260803_002_canonical_publication.receipt.json"
)
_SUPERSEDED_RECEIPT = (
    _FIXTURES / "source_pilot_20260803_002_superseded_publication.receipt.json"
)


def _binding(adjudication: dict[str, object], name: str) -> dict[str, object]:
    value = adjudication[name]
    if name == "superseded":
        assert isinstance(value, list) and len(value) == 1
        value = value[0]
    assert isinstance(value, dict)
    return value


def test_exact_canonical_source_pilot_publication_is_selected(
    tmp_path: Path,
) -> None:
    adjudication = load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH)
    authoritative = _binding(adjudication, "authoritative")
    receipt_binding = authoritative["publication_receipt"]
    assert isinstance(receipt_binding, dict)

    assert authoritative["artifact_set_sha256"] == AUTHORITATIVE_ARTIFACT_SET_SHA256
    assert authoritative["archive"]["generation"] == AUTHORITATIVE_ARCHIVE_GENERATION
    assert receipt_binding["generation"] == AUTHORITATIVE_RECEIPT_GENERATION
    assert (
        hashlib.sha256(_CANONICAL_RECEIPT.read_bytes()).hexdigest()
        == receipt_binding["sha256"]
    )
    selection = select_source_pilot_publication(
        adjudication_path=DEFAULT_ADJUDICATION_PATH,
        publication_receipt_path=_CANONICAL_RECEIPT,
        publication_receipt_uri=str(receipt_binding["uri"]),
        publication_receipt_generation=AUTHORITATIVE_RECEIPT_GENERATION,
    )

    assert selection["schema"] == SOURCE_PILOT_SELECTION_SCHEMA
    assert selection["status"] == "selected"
    assert selection["training_ready"] is False
    assert selection["adjudication_sha256"] == EXPECTED_ADJUDICATION_SHA256
    assert selection["selected"] == authoritative
    declared_selection_sha256 = selection.pop("selection_sha256")
    assert canonical_sha256(selection) == declared_selection_sha256
    selection["selection_sha256"] = declared_selection_sha256

    output = tmp_path / "selection.json"
    write_selection_receipt(output, selection)
    write_selection_receipt(output, selection)
    assert json.loads(output.read_text(encoding="utf-8")) == selection
    assert load_source_pilot_selection(output) == selection


def test_exact_c88ace_source_pilot_publication_is_explicitly_superseded() -> None:
    adjudication = load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH)
    superseded = _binding(adjudication, "superseded")
    receipt_binding = superseded["publication_receipt"]
    assert isinstance(receipt_binding, dict)

    assert superseded["artifact_set_sha256"] == SUPERSEDED_ARTIFACT_SET_SHA256
    assert superseded["reason"] == "unreceipted_sqlite_sidecars"
    assert hashlib.sha256(_SUPERSEDED_RECEIPT.read_bytes()).hexdigest() == (
        receipt_binding["sha256"]
    )
    with pytest.raises(
        SupersededPublicationError,
        match="c88ace50.*explicitly superseded.*sqlite-shm.*sqlite-wal",
    ):
        select_source_pilot_publication(
            adjudication_path=DEFAULT_ADJUDICATION_PATH,
            publication_receipt_path=_SUPERSEDED_RECEIPT,
            publication_receipt_uri=str(receipt_binding["uri"]),
            publication_receipt_generation=str(receipt_binding["generation"]),
        )


@pytest.mark.parametrize(
    ("uri_suffix", "generation"),
    [
        ("", "1785783193773476"),
        (".drift", AUTHORITATIVE_RECEIPT_GENERATION),
    ],
)
def test_unadjudicated_receipt_observation_fails_closed(
    uri_suffix: str,
    generation: str,
) -> None:
    adjudication = load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH)
    authoritative = _binding(adjudication, "authoritative")
    receipt_binding = authoritative["publication_receipt"]
    assert isinstance(receipt_binding, dict)

    with pytest.raises(ContractError, match="neither the exact authoritative"):
        select_source_pilot_publication(
            adjudication_path=DEFAULT_ADJUDICATION_PATH,
            publication_receipt_path=_CANONICAL_RECEIPT,
            publication_receipt_uri=f"{receipt_binding['uri']}{uri_suffix}",
            publication_receipt_generation=generation,
        )


def test_recomputed_but_changed_adjudication_is_not_authoritative(
    tmp_path: Path,
) -> None:
    adjudication = json.loads(DEFAULT_ADJUDICATION_PATH.read_text(encoding="utf-8"))
    adjudication["authoritative"]["archive"]["generation"] = "1785783190042980"
    digest_payload = dict(adjudication)
    digest_payload.pop("adjudication_sha256")
    adjudication["adjudication_sha256"] = canonical_sha256(digest_payload)
    path = tmp_path / "rewritten-adjudication.json"
    atomic_write_json(path, adjudication)

    with pytest.raises(ContractError, match="adjudication identity drifted"):
        load_source_pilot_adjudication(path)


def test_recomputed_but_changed_selection_is_rejected(tmp_path: Path) -> None:
    adjudication = load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH)
    authoritative = _binding(adjudication, "authoritative")
    receipt_binding = authoritative["publication_receipt"]
    assert isinstance(receipt_binding, dict)
    selection = select_source_pilot_publication(
        adjudication_path=DEFAULT_ADJUDICATION_PATH,
        publication_receipt_path=_CANONICAL_RECEIPT,
        publication_receipt_uri=str(receipt_binding["uri"]),
        publication_receipt_generation=str(receipt_binding["generation"]),
    )
    selection["selected"]["archive"]["generation"] = "1785783190042980"
    selection.pop("selection_sha256")
    selection["selection_sha256"] = canonical_sha256(selection)
    path = tmp_path / "rewritten-selection.json"
    atomic_write_json(path, selection)

    with pytest.raises(ContractError, match="differs from adjudication"):
        load_source_pilot_selection(path)


def test_cli_writes_selection_and_returns_distinct_superseded_exit(
    tmp_path: Path,
) -> None:
    adjudication = load_source_pilot_adjudication(DEFAULT_ADJUDICATION_PATH)
    authoritative = _binding(adjudication, "authoritative")
    authoritative_receipt = authoritative["publication_receipt"]
    assert isinstance(authoritative_receipt, dict)
    output = tmp_path / "selection.json"
    script = (
        Path(__file__).parents[1]
        / "scripts/distributed_data_prep/adjudicate_source_pilot.py"
    )
    accepted = subprocess.run(
        [
            sys.executable,
            str(script),
            "--publication-receipt",
            str(_CANONICAL_RECEIPT),
            "--publication-receipt-uri",
            str(authoritative_receipt["uri"]),
            "--publication-receipt-generation",
            str(authoritative_receipt["generation"]),
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert accepted.returncode == 0, accepted.stderr
    assert output.is_file()

    superseded = _binding(adjudication, "superseded")
    superseded_receipt = superseded["publication_receipt"]
    assert isinstance(superseded_receipt, dict)
    rejected = subprocess.run(
        [
            sys.executable,
            str(script),
            "--publication-receipt",
            str(_SUPERSEDED_RECEIPT),
            "--publication-receipt-uri",
            str(superseded_receipt["uri"]),
            "--publication-receipt-generation",
            str(superseded_receipt["generation"]),
            "--output",
            str(tmp_path / "must-not-exist.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert rejected.returncode == 3
    assert "explicitly superseded" in rejected.stderr
    assert not (tmp_path / "must-not-exist.json").exists()
