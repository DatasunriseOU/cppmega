"""Stable tokenizer fingerprints for dataset/consumer compatibility checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def tokenizer_fingerprint(tokenizer_or_path: Any) -> str:
    """Hash tokenizer vocab, merges, and special-token map as canonical JSON."""

    tokenizer_json = _tokenizer_json(tokenizer_or_path)
    payload = {
        "model": {
            "type": tokenizer_json.get("model", {}).get("type"),
            "unk_token": tokenizer_json.get("model", {}).get("unk_token"),
            "vocab": tokenizer_json.get("model", {}).get("vocab", {}),
            "merges": tokenizer_json.get("model", {}).get("merges", []),
        },
        "special_tokens": sorted(
            (
                int(token.get("id", -1)),
                str(token.get("content", "")),
                bool(token.get("special", False)),
            )
            for token in tokenizer_json.get("added_tokens", [])
            if bool(token.get("special", False))
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _tokenizer_json(tokenizer_or_path: Any) -> dict[str, Any]:
    path = getattr(tokenizer_or_path, "path", tokenizer_or_path)
    if isinstance(path, (str, os.PathLike)):
        return json.loads(Path(path).read_text(encoding="utf-8"))

    tokenizer = getattr(tokenizer_or_path, "_tokenizer", tokenizer_or_path)
    if hasattr(tokenizer, "to_str"):
        return json.loads(tokenizer.to_str())

    if hasattr(tokenizer, "get_vocab"):
        return {
            "model": {
                "type": None,
                "unk_token": None,
                "vocab": tokenizer.get_vocab(),
                "merges": [],
            },
            "added_tokens": [],
        }

    raise TypeError("tokenizer_fingerprint requires a tokenizer path or object")
