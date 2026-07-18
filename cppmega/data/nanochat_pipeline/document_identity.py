"""Portable stable identities for logical source documents."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


_EXPLICIT_DOC_ID_COLUMNS = (
    "source_doc_id",
    "source_document_id",
    "document_id",
    "doc_id",
)
_PROVENANCE_SIGNATURE_COLUMNS = (
    "repo_stable_id",
    "filepath_stable_id",
    "commit_hash",
    "file_local_commit_index",
)


def stable_doc_signature(row: Mapping[str, Any]) -> str:
    """Return a deterministic signature for one logical document."""

    for column in _EXPLICIT_DOC_ID_COLUMNS:
        value = row.get(column)
        if value is not None:
            return f"{column}:{value}"

    provenance = tuple(row.get(column) for column in _PROVENANCE_SIGNATURE_COLUMNS)
    if any(value is not None for value in provenance):
        return "provenance:" + "\0".join(
            "" if value is None else str(value) for value in provenance
        )

    text = row.get("text")
    if isinstance(text, str) and text:
        return _text_signature(row)

    token_ids = row.get("token_ids")
    if token_ids is not None:
        encoded = json.dumps(
            [int(token_id) for token_id in token_ids],
            separators=(",", ":"),
        ).encode("ascii")
        return "token_ids_sha256:" + hashlib.sha256(encoded).hexdigest()

    return _text_signature(row)


def _text_signature(row: Mapping[str, Any]) -> str:
    text = row.get("text", "")
    if not isinstance(text, str):
        text = repr(text)
    return "text_sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


__all__ = ["stable_doc_signature"]
