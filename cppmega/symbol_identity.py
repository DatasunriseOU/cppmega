"""Corpus-wide v3 symbol identity projection and collision registry."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import hashlib


SYMBOL_IDENTITY_SCHEMA_VERSION = 3
SYMBOL_IDENTITY_SCHEMA_METADATA_KEY = "cppmega.symbol_identity_schema_version"
SYMBOL_IDENTITIES_COLUMN = "symbol_identities"
SYMBOL_ID_MAX = (1 << 64) - 1
_SYMBOL_ID_HASH_DOMAIN = b"cppmega.symbol-id.v3\0"


class SymbolIdentityError(RuntimeError):
    """Raised when a v3 ID/key claim is inconsistent or colliding."""


def compute_symbol_id(symbol_key: str) -> int:
    if not symbol_key:
        return 0
    digest = hashlib.sha256(
        _SYMBOL_ID_HASH_DOMAIN + symbol_key.encode("utf-8", errors="strict")
    ).digest()
    symbol_id = int.from_bytes(digest[:8], byteorder="big", signed=False)
    if symbol_id == 0:
        raise SymbolIdentityError(
            "canonical symbol key hashed to reserved ID 0; refusing to alias it"
        )
    return symbol_id


class SymbolIdentityRegistry:
    def __init__(self) -> None:
        self.keys_by_id: dict[int, str] = {}
        self.ids_by_key: dict[str, int] = {}
        self.sources_by_id: dict[int, str] = {}

    def register(
        self,
        symbol_key: str,
        *,
        symbol_id: int | None = None,
        source: str,
    ) -> int:
        if not isinstance(symbol_key, str) or not symbol_key:
            raise SymbolIdentityError(f"{source}: symbol_key must be non-empty")
        claimed_id = compute_symbol_id(symbol_key) if symbol_id is None else int(symbol_id)
        if not 0 < claimed_id <= SYMBOL_ID_MAX:
            raise SymbolIdentityError(
                f"{source}: symbol_id must be in [1, {SYMBOL_ID_MAX}], got {claimed_id}"
            )
        existing_key = self.keys_by_id.get(claimed_id)
        if existing_key is not None and existing_key != symbol_key:
            raise SymbolIdentityError(
                "canonical symbol ID collision: "
                f"id={claimed_id} first={existing_key!r} "
                f"({self.sources_by_id.get(claimed_id, 'unknown source')}) "
                f"second={symbol_key!r} ({source})"
            )
        existing_id = self.ids_by_key.get(symbol_key)
        if existing_id is not None and existing_id != claimed_id:
            raise SymbolIdentityError(
                f"{source}: canonical key {symbol_key!r} maps to both "
                f"{existing_id} and {claimed_id}"
            )
        expected_id = compute_symbol_id(symbol_key)
        if claimed_id != expected_id:
            raise SymbolIdentityError(
                f"{source}: symbol_id {claimed_id} does not match v3 ID "
                f"{expected_id} for {symbol_key!r}"
            )
        self.keys_by_id[claimed_id] = symbol_key
        self.ids_by_key[symbol_key] = claimed_id
        self.sources_by_id.setdefault(claimed_id, source)
        return claimed_id

    def register_records(self, records: object, *, source: str) -> None:
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise SymbolIdentityError(
                f"{source}: {SYMBOL_IDENTITIES_COLUMN} must be a list"
            )
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise SymbolIdentityError(
                    f"{source}: {SYMBOL_IDENTITIES_COLUMN}[{index}] must be an object"
                )
            symbol_key = record.get("symbol_key")
            symbol_id = record.get("symbol_id")
            if not isinstance(symbol_key, str) or symbol_id is None:
                raise SymbolIdentityError(
                    f"{source}: identity record {index} requires symbol_id and symbol_key"
                )
            self.register(symbol_key, symbol_id=int(symbol_id), source=source)

    def require_ids(self, symbol_ids: Iterable[int], *, source: str) -> None:
        missing = sorted(
            {
                int(symbol_id)
                for symbol_id in symbol_ids
                if int(symbol_id) and int(symbol_id) not in self.keys_by_id
            }
        )
        if missing:
            raise SymbolIdentityError(
                f"{source}: semantic symbol IDs have no canonical claims: {missing[:8]}"
            )


__all__ = [
    "SYMBOL_IDENTITIES_COLUMN",
    "SYMBOL_IDENTITY_SCHEMA_METADATA_KEY",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
    "SYMBOL_ID_MAX",
    "SymbolIdentityError",
    "SymbolIdentityRegistry",
    "compute_symbol_id",
]
