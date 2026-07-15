"""Compatibility re-export of the canonical symbol identity contract."""

from cppmega.symbol_identity import (
    REPO_FILE_LOCATION_IDENTITY_PREFIX,
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SYMBOL_ID_MAX,
    RepoFileLocationIdentity,
    ResolvedProjectIdentity,
    SymbolIdentityError,
    SymbolIdentityRegistry,
    compute_symbol_id,
    is_repo_file_location_identity,
    parse_repo_file_location_identity,
    require_project_identity,
    resolve_remote_project_identity,
)

__all__ = [
    "REPO_FILE_LOCATION_IDENTITY_PREFIX",
    "SYMBOL_IDENTITIES_COLUMN",
    "SYMBOL_IDENTITY_SCHEMA_METADATA_KEY",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
    "SYMBOL_ID_MAX",
    "ResolvedProjectIdentity",
    "RepoFileLocationIdentity",
    "SymbolIdentityError",
    "SymbolIdentityRegistry",
    "compute_symbol_id",
    "is_repo_file_location_identity",
    "parse_repo_file_location_identity",
    "require_project_identity",
    "resolve_remote_project_identity",
]
