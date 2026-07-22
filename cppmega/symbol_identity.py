"""Corpus-wide symbol identity ID contract.

The canonical clang identity key remains the source of truth. Numeric IDs are
opaque unsigned 64-bit projections used by dense training sidecars. Every
serialized corpus row carries the corresponding ID-to-key claims so aggregators
can detect a collision and fail closed instead of aliasing supervision.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import ipaddress
import posixpath
import re
import unicodedata
from urllib.parse import quote, unquote_to_bytes, urlsplit


SYMBOL_IDENTITY_SCHEMA_VERSION = 3
SYMBOL_IDENTITY_SCHEMA_METADATA_KEY = "cppmega.symbol_identity_schema_version"
SYMBOL_IDENTITIES_COLUMN = "symbol_identities"
SYMBOL_ID_MAX = (1 << 64) - 1
_SYMBOL_ID_HASH_DOMAIN = b"cppmega.symbol-id.v3\0"
REPO_FILE_LOCATION_IDENTITY_PREFIX = "repo_file_location:"
EXTERNAL_PROVIDER_FILE_PREFIX = "@provider/"
EXTERNAL_PROVIDER_PROJECTS = {
    "libc++": "llvm/llvm-project",
    "libstdc++": "gcc-mirror/gcc",
    "msvc-stl": "microsoft/STL",
    "boost": "boostorg/boost",
}


class SymbolIdentityError(RuntimeError):
    """Raised when a symbol identity claim violates the v3 contract."""


@dataclass(frozen=True)
class ResolvedProjectIdentity:
    """Canonical symbol namespace plus an optional GitHub PR lookup key."""

    project_identity: str
    owner_repo: str | None


@dataclass(frozen=True)
class RepoFileLocationIdentity:
    """Validated fields carried by an explicit repository-location identity."""

    project: str
    file: str
    line: int
    column: int
    kind: str
    qname: str


@dataclass(frozen=True)
class UsrIdentity:
    """Parsed clang USR identity, optionally scoped to one repository file."""

    project: str
    usr: str
    file: str
    canonical_signature: str


_IDENTITY_COMPONENT_SAFE = "-._~"
_INVALID_PERCENT_ESCAPE_RE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_ENCODED_PATH_SEPARATOR_RE = re.compile(r"%(?:2[fF]|5[cC])")
_WINDOWS_LOCAL_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")
_REMOTE_SCHEME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_SCP_REMOTE_RE = re.compile(
    r"^(?:[^@/:\s]+@)?(?P<host>\[[^\]]+\]|[^/:\s]+):(?P<path>.+)$"
)
_HOST_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$")
_GITHUB_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_GITHUB_HOSTS = frozenset({"github.com", "ssh.github.com", "www.github.com"})
_DEFAULT_REMOTE_PORTS = {
    "git": 9418,
    "http": 80,
    "https": 443,
    "ssh": 22,
}


def _contains_unsafe_path_text(value: str) -> bool:
    return any(
        char == "\\" or char.isspace() or ord(char) < 32 or ord(char) == 127
        for char in value
    )


def _decode_percent_encoded(value: str, *, source: str) -> str:
    if _INVALID_PERCENT_ESCAPE_RE.search(value):
        raise SymbolIdentityError(
            f"{source}: malformed percent escape in project path {value!r}"
        )
    try:
        decoded = unquote_to_bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SymbolIdentityError(
            f"{source}: project path is not valid UTF-8"
        ) from exc
    normalized = unicodedata.normalize("NFC", decoded)
    if normalized != decoded:
        raise SymbolIdentityError(
            f"{source}: project path must use canonical NFC Unicode"
        )
    return decoded


def _validate_project_path(path: str, *, source: str) -> tuple[str, ...]:
    if not path or _contains_unsafe_path_text(path):
        raise SymbolIdentityError(
            f"{source}: project path must be non-empty and contain no whitespace, "
            "control characters, or backslashes"
        )
    parts = tuple(path.split("/"))
    if any(not part or part in {".", ".."} for part in parts):
        raise SymbolIdentityError(
            f"{source}: project path contains an empty or traversal segment: {path!r}"
        )
    return parts


def _decode_identity_component(component: str, *, source: str) -> str:
    decoded = _decode_percent_encoded(component, source=source)
    if quote(decoded, safe=_IDENTITY_COMPONENT_SAFE) != component:
        raise SymbolIdentityError(
            f"{source}: project identity component is not canonically encoded: "
            f"{component!r}"
        )
    return decoded


def require_project_identity(value: object, *, source: str) -> str:
    """Validate one canonical, exact-one-slash project identity.

    GitHub identities retain their historical ``owner/repo`` form. Other
    forges use ``host/percent-encoded-path`` so an arbitrary-depth remote path
    remains lossless without adding literal slashes to the symbol namespace.
    """

    if not isinstance(value, str) or value != value.strip():
        raise SymbolIdentityError(
            f"{source}: project identity must be an exact namespace/project "
            f"string, got {value!r}"
        )
    if value.count("/") != 1:
        raise SymbolIdentityError(
            f"{source}: project identity must contain exactly one slash, got {value!r}"
        )
    namespace, project = value.split("/", 1)
    if not namespace or not project:
        raise SymbolIdentityError(
            f"{source}: project identity components must be non-empty, got {value!r}"
        )
    decoded_namespace = _decode_identity_component(
        namespace, source=f"{source}:namespace"
    )
    decoded_project = _decode_identity_component(
        project, source=f"{source}:project"
    )
    if (
        decoded_namespace in {".", ".."}
        or "/" in decoded_namespace
        or _contains_unsafe_path_text(decoded_namespace)
    ):
        raise SymbolIdentityError(
            f"{source}: project identity has an unsafe namespace: {value!r}"
        )
    project_parts = _validate_project_path(decoded_project, source=source)
    if project_parts[-1].lower().endswith(".git"):
        raise SymbolIdentityError(
            f"{source}: project identity must be canonical without .git, got {value!r}"
        )
    return value


def parse_repo_file_location_identity(
    value: object,
    *,
    source: str = "symbol identity",
) -> RepoFileLocationIdentity:
    """Parse and validate the explicit no-USR/no-signature identity form.

    This identity is deliberately distinct from a signature fallback. It is
    valid only when project, repository-relative file and declaration location
    are all explicit, so downstream stores can accept it without pretending a
    qname or an exception sentinel is a semantic signature.
    """

    if not isinstance(value, str) or not value.startswith(
        REPO_FILE_LOCATION_IDENTITY_PREFIX
    ):
        raise SymbolIdentityError(
            f"{source}: expected {REPO_FILE_LOCATION_IDENTITY_PREFIX!r} identity"
        )
    payload = value.removeprefix(REPO_FILE_LOCATION_IDENTITY_PREFIX)
    parts = payload.split("\x1f")
    if len(parts) not in {6, 7}:
        raise SymbolIdentityError(
            f"{source}: repo_file_location identity has {len(parts)} fields; "
            "expected 6 or 7"
        )
    expected_keys = ["schema", "project", "file", "line"]
    if len(parts) == 7:
        expected_keys.append("column")
    expected_keys.extend(("kind", "qname"))
    fields: dict[str, str] = {}
    for part, expected_key in zip(parts, expected_keys, strict=True):
        key, separator, field_value = part.partition("=")
        if separator != "=" or key != expected_key or not field_value:
            raise SymbolIdentityError(
                f"{source}: repo_file_location identity requires ordered non-empty "
                f"field {expected_key!r}"
            )
        if any(ord(char) < 32 or ord(char) == 127 for char in field_value):
            raise SymbolIdentityError(
                f"{source}: repo_file_location field {expected_key!r} contains "
                "control characters"
            )
        fields[key] = field_value

    expected_schema = f"v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
    if fields["schema"] != expected_schema:
        raise SymbolIdentityError(
            f"{source}: repo_file_location identity schema must be {expected_schema}"
        )
    project = require_project_identity(
        fields["project"], source=f"{source}:repo_file_location project"
    )
    file = fields["file"]
    normalized_file = posixpath.normpath(file)
    if (
        file != normalized_file
        or file.startswith("/")
        or _WINDOWS_LOCAL_PATH_RE.match(file)
        or "\\" in file
        or any(part in {"", ".", ".."} for part in file.split("/"))
    ):
        raise SymbolIdentityError(
            f"{source}: repo_file_location file must be canonical and "
            f"repository-relative, got {file!r}"
        )

    def parse_positive_int(field: str, *, required: bool) -> int:
        raw = fields.get(field, "")
        if not raw:
            return 0
        if not raw.isascii() or not raw.isdecimal():
            raise SymbolIdentityError(
                f"{source}: repo_file_location {field} must be a decimal integer"
            )
        parsed = int(raw)
        if str(parsed) != raw or parsed < (1 if required else 0):
            qualifier = "positive" if required else "non-negative"
            raise SymbolIdentityError(
                f"{source}: repo_file_location {field} must be canonical {qualifier} "
                "decimal"
            )
        return parsed

    return RepoFileLocationIdentity(
        project=project,
        file=file,
        line=parse_positive_int("line", required=True),
        column=parse_positive_int("column", required=True),
        kind=fields["kind"],
        qname=fields["qname"],
    )


def is_repo_file_location_identity(value: object) -> bool:
    """Return whether ``value`` is a valid explicit location identity.

    A malformed value using the reserved prefix raises instead of being treated
    as an unrelated key, preserving fail-closed validation at storage boundaries.
    """

    if not isinstance(value, str) or not value.startswith(
        REPO_FILE_LOCATION_IDENTITY_PREFIX
    ):
        return False
    parse_repo_file_location_identity(value)
    return True


def _validate_usr_text(value: object, *, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or "\x1f" in value
        or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127
            for char in value
        )
    ):
        raise SymbolIdentityError(f"{source}: clang USR is missing or unsafe")
    return value


def _validate_usr_signature(value: object, *, source: str) -> str:
    signature = " ".join(str(value or "").split())
    if not signature or "\x1f" in signature:
        raise SymbolIdentityError(
            f"{source}: canonical signature is required for scoped USR identity"
        )
    return signature


def _validate_usr_file(value: object, *, source: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != posixpath.normpath(value)
        or value.startswith("/")
        or _WINDOWS_LOCAL_PATH_RE.match(value)
        or "\\" in value
        or any(part in {"", ".", ".."} for part in value.split("/"))
        or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127
            for char in value
        )
    ):
        raise SymbolIdentityError(
            f"{source}: scoped USR file must be canonical and repository-relative"
        )
    return value


def canonical_usr_identity(
    *,
    usr: object,
    project: object,
    file: object | None = None,
    canonical_signature: object = "",
    source: str = "clang USR identity",
) -> str:
    """Return a project-bound clang USR key.

    The historical three-field form is retained for externally-linkable
    symbols. File-local or otherwise explicitly scoped symbols use the
    five-field form and bind the USR to the repository file and canonical
    signature. A qname is never part of either identity form.
    """

    canonical_project = require_project_identity(project, source=f"{source}:project")
    canonical_usr = _validate_usr_text(usr, source=source)
    if file is None:
        return (
            f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
            f"project={canonical_project}\x1fusr={canonical_usr}"
        )
    canonical_file = _validate_usr_file(file, source=f"{source}:file")
    signature = _validate_usr_signature(
        canonical_signature,
        source=f"{source}:signature",
    )
    encoded_signature = quote(signature, safe=_IDENTITY_COMPONENT_SAFE)
    return (
        f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
        f"project={canonical_project}\x1ffile={canonical_file}\x1f"
        f"sig={encoded_signature}\x1fusr={canonical_usr}"
    )


def parse_usr_identity(
    value: object,
    *,
    source: str = "clang USR identity",
) -> UsrIdentity:
    """Parse either the historical or file-scoped canonical USR form."""

    if not isinstance(value, str) or not value.startswith("usr:"):
        raise SymbolIdentityError(f"{source}: expected a usr: identity")
    parts = value.removeprefix("usr:").split("\x1f")
    if len(parts) == 3:
        expected = ("schema", "project", "usr")
    elif len(parts) == 5:
        expected = ("schema", "project", "file", "sig", "usr")
    else:
        raise SymbolIdentityError(
            f"{source}: USR identity has {len(parts)} fields; expected 3 or 5"
        )
    fields: dict[str, str] = {}
    for part, key in zip(parts, expected, strict=True):
        actual_key, separator, field_value = part.partition("=")
        if separator != "=" or actual_key != key or not field_value:
            raise SymbolIdentityError(
                f"{source}: USR identity requires ordered non-empty field {key!r}"
            )
        fields[key] = field_value
    if fields["schema"] != f"v{SYMBOL_IDENTITY_SCHEMA_VERSION}":
        raise SymbolIdentityError(
            f"{source}: USR identity schema must be v{SYMBOL_IDENTITY_SCHEMA_VERSION}"
        )
    project = require_project_identity(fields["project"], source=f"{source}:project")
    usr = _validate_usr_text(fields["usr"], source=source)
    if len(parts) == 3:
        return UsrIdentity(project, usr, "", "")
    file = _validate_usr_file(fields["file"], source=f"{source}:file")
    try:
        signature = unquote_to_bytes(fields["sig"]).decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise SymbolIdentityError(
            f"{source}: USR identity signature is not valid UTF-8"
        ) from exc
    signature = _validate_usr_signature(signature, source=f"{source}:signature")
    if quote(signature, safe=_IDENTITY_COMPONENT_SAFE) != fields["sig"]:
        raise SymbolIdentityError(
            f"{source}: USR identity signature is not canonically encoded"
        )
    expected = canonical_usr_identity(
        usr=usr,
        project=project,
        file=file,
        canonical_signature=signature,
        source=source,
    )
    if value != expected:
        raise SymbolIdentityError(f"{source}: USR identity is not canonical")
    return UsrIdentity(project, usr, file, signature)


def is_scoped_usr_identity(value: object) -> bool:
    """Return whether a valid USR identity carries repository-file scope."""

    return bool(parse_usr_identity(value).file)


def external_provider_project(
    provider: object,
    *,
    source: str = "external symbol provider",
) -> str:
    """Return the authoritative project namespace for a trusted provider."""

    if (
        not isinstance(provider, str)
        or provider != provider.strip()
        or provider not in EXTERNAL_PROVIDER_PROJECTS
    ):
        raise SymbolIdentityError(
            f"{source}: unsupported external symbol provider {provider!r}"
        )
    return EXTERNAL_PROVIDER_PROJECTS[provider]


def canonical_external_provider_file(
    provider: object,
    include_provenance: object,
    *,
    source: str = "external symbol provider file",
) -> str:
    """Return a canonical provider-owned include identity."""

    external_provider_project(provider, source=source)
    normalized_provider = str(provider)
    if (
        not isinstance(include_provenance, str)
        or not include_provenance
        or include_provenance != include_provenance.strip()
    ):
        raise SymbolIdentityError(
            f"{source}: include provenance must be a non-empty canonical path"
        )
    include = include_provenance
    if (
        posixpath.normpath(include) != include
        or include.startswith("/")
        or _WINDOWS_LOCAL_PATH_RE.match(include)
        or "\\" in include
        or any(part in {"", ".", ".."} for part in include.split("/"))
        or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127
            for char in include
        )
    ):
        raise SymbolIdentityError(
            f"{source}: include provenance is not canonical: {include!r}"
        )
    return f"{EXTERNAL_PROVIDER_FILE_PREFIX}{normalized_provider}/{include}"


def parse_external_provider_file(
    value: object,
    *,
    source: str = "external symbol provider file",
) -> tuple[str, str]:
    """Parse and validate a canonical ``@provider/...`` identity."""

    if not isinstance(value, str) or not value.startswith(
        EXTERNAL_PROVIDER_FILE_PREFIX
    ):
        raise SymbolIdentityError(
            f"{source}: expected an {EXTERNAL_PROVIDER_FILE_PREFIX!r} path"
        )
    payload = value.removeprefix(EXTERNAL_PROVIDER_FILE_PREFIX)
    provider, separator, include = payload.partition("/")
    if not separator:
        raise SymbolIdentityError(f"{source}: provider include path is missing")
    expected = canonical_external_provider_file(
        provider,
        include,
        source=source,
    )
    if value != expected:
        raise SymbolIdentityError(
            f"{source}: provider include path is not canonical: {value!r}"
        )
    return provider, include


def canonical_external_usr_identity(
    *,
    usr: object,
    canonical_signature: object,
    provider: object,
    include_provenance: object,
    project: object | None = None,
    source: str = "external USR identity",
) -> str:
    """Bind an external clang USR to provider, include, and overload signature."""

    provider_project = external_provider_project(provider, source=source)
    provider_name = str(provider)
    provider_file = canonical_external_provider_file(
        provider,
        include_provenance,
        source=source,
    )
    if project is not None:
        claimed_project = require_project_identity(project, source=f"{source}:project")
        if claimed_project != provider_project:
            raise SymbolIdentityError(
                f"{source}: provider {provider!r} belongs to {provider_project!r}, "
                f"not {claimed_project!r}"
            )
    if (
        not isinstance(usr, str)
        or not usr
        or usr != usr.strip()
        or "\x1f" in usr
        or any(
            char.isspace() or ord(char) < 32 or ord(char) == 127
            for char in usr
        )
    ):
        raise SymbolIdentityError(f"{source}: clang USR is missing or unsafe")
    signature = " ".join(str(canonical_signature or "").split())
    if not signature or "\x1f" in signature:
        raise SymbolIdentityError(
            f"{source}: canonical signature is required for external USR identity"
        )
    _provider, include = parse_external_provider_file(provider_file, source=source)
    encoded_signature = quote(signature, safe=_IDENTITY_COMPONENT_SAFE)
    return (
        f"usr:schema=v{SYMBOL_IDENTITY_SCHEMA_VERSION}\x1f"
        f"project={provider_project}\x1fprovider={provider_name}\x1f"
        f"include={include}\x1fsig={encoded_signature}\x1fusr={usr}"
    )


def _normalize_remote_host(host: str, *, source: str) -> str:
    raw_host = host.strip().rstrip(".")
    if not raw_host or _contains_unsafe_path_text(raw_host) or "%" in raw_host:
        raise SymbolIdentityError(f"{source}: remote host is unsafe or empty")
    try:
        ip = ipaddress.ip_address(raw_host)
    except ValueError:
        try:
            normalized = raw_host.encode("idna").decode("ascii").lower()
        except UnicodeError as exc:
            raise SymbolIdentityError(
                f"{source}: remote host is not valid IDNA: {raw_host!r}"
            ) from exc
        labels = normalized.split(".")
        if (
            len(normalized) > 253
            or any(not _HOST_LABEL_RE.fullmatch(label) for label in labels)
        ):
            raise SymbolIdentityError(
                f"{source}: remote host is not canonical: {raw_host!r}"
            )
        return normalized
    return ip.compressed.lower()


def _split_remote(remote_url: str, *, source: str) -> tuple[str, int | None, str, str]:
    if _WINDOWS_LOCAL_PATH_RE.match(remote_url):
        raise SymbolIdentityError(
            f"{source}: local filesystem path is not a network forge remote"
        )
    if _REMOTE_SCHEME_RE.match(remote_url):
        parsed = urlsplit(remote_url)
        if parsed.scheme.lower() == "file" or not parsed.netloc or not parsed.hostname:
            raise SymbolIdentityError(
                f"{source}: remote must identify a network forge host"
            )
        if parsed.query or parsed.fragment:
            raise SymbolIdentityError(
                f"{source}: remote query and fragment are not project identity"
            )
        try:
            port = parsed.port
        except ValueError as exc:
            raise SymbolIdentityError(f"{source}: remote port is invalid") from exc
        return parsed.hostname, port, parsed.path, parsed.scheme.lower()

    scp_match = _SCP_REMOTE_RE.fullmatch(remote_url)
    if scp_match:
        host = scp_match.group("host")
        if host.startswith("[") and host.endswith("]"):
            host = host[1:-1]
        raw_path = scp_match.group("path")
        if "?" in raw_path or "#" in raw_path:
            raise SymbolIdentityError(
                f"{source}: remote query and fragment are not project identity"
            )
        return host, None, raw_path, "ssh"

    if "/" in remote_url:
        host, raw_path = remote_url.split("/", 1)
        if "." in host and host and raw_path:
            if "?" in raw_path or "#" in raw_path:
                raise SymbolIdentityError(
                    f"{source}: remote query and fragment are not project identity"
                )
            return host, None, raw_path, ""

    raise SymbolIdentityError(
        f"{source}: remote does not identify a canonical network forge project"
    )


def _normalize_remote_path(raw_path: str, *, source: str) -> tuple[str, ...]:
    if raw_path != raw_path.strip():
        raise SymbolIdentityError(
            f"{source}: remote project path has surrounding whitespace"
        )
    path = raw_path
    if path.startswith("/"):
        path = path[1:]
    if path.endswith("/"):
        path = path[:-1]
    if not path or path.startswith("/") or path.endswith("/"):
        raise SymbolIdentityError(
            f"{source}: remote project path has ambiguous repeated slashes"
        )
    if _ENCODED_PATH_SEPARATOR_RE.search(path):
        raise SymbolIdentityError(
            f"{source}: remote project path has an ambiguous encoded separator"
        )
    decoded = _decode_percent_encoded(path, source=source)
    parts = list(_validate_project_path(decoded, source=source))
    if parts[-1].lower().endswith(".git"):
        parts[-1] = parts[-1][:-4]
        if not parts[-1] or parts[-1] in {".", ".."}:
            raise SymbolIdentityError(
                f"{source}: remote project path is empty after removing .git"
            )
    return tuple(parts)


def resolve_remote_project_identity(
    remote_url: object,
    *,
    source: str,
) -> ResolvedProjectIdentity:
    """Resolve a network git remote to a collision-safe project identity.

    The remote is authoritative. Local paths and malformed/ambiguous remotes
    fail closed so callers cannot invent an identity from a clone directory.
    """

    if not isinstance(remote_url, str) or not remote_url.strip():
        raise SymbolIdentityError(f"{source}: remote URL must be a non-empty string")
    remote = remote_url.strip()
    host, port, raw_path, scheme = _split_remote(remote, source=source)
    canonical_host = _normalize_remote_host(host, source=source)
    path_parts = _normalize_remote_path(raw_path, source=source)

    if canonical_host in _GITHUB_HOSTS:
        if len(path_parts) != 2:
            raise SymbolIdentityError(
                f"{source}: GitHub remote must have exactly owner/repo path segments"
            )
        owner, repo = path_parts
        if (
            not _GITHUB_COMPONENT_RE.fullmatch(owner)
            or not _GITHUB_COMPONENT_RE.fullmatch(repo)
        ):
            raise SymbolIdentityError(
                f"{source}: GitHub owner/repo contains unsupported characters"
            )
        owner_repo = require_project_identity(
            f"{owner}/{repo}", source=f"{source}:GitHub"
        )
        return ResolvedProjectIdentity(
            project_identity=owner_repo,
            owner_repo=owner_repo,
        )

    authority = canonical_host
    if port is not None and port != _DEFAULT_REMOTE_PORTS.get(scheme):
        authority = f"{authority}:{port}"
    project_identity = require_project_identity(
        f"{quote(authority, safe=_IDENTITY_COMPONENT_SAFE)}/"
        f"{quote('/'.join(path_parts), safe=_IDENTITY_COMPONENT_SAFE)}",
        source=source,
    )
    return ResolvedProjectIdentity(
        project_identity=project_identity,
        owner_repo=None,
    )


def compute_symbol_id(symbol_key: str) -> int:
    """Return the stable unsigned 64-bit ID for one canonical identity key."""

    if not symbol_key:
        return 0
    digest = hashlib.sha256(
        _SYMBOL_ID_HASH_DOMAIN + symbol_key.encode("utf-8", errors="strict")
    ).digest()
    symbol_id = int.from_bytes(digest[:8], byteorder="big", signed=False)
    if symbol_id == 0:
        raise SymbolIdentityError(
            "canonical symbol key hashed to reserved ID 0; refusing to emit "
            f"semantic supervision for {symbol_key!r}"
        )
    return symbol_id


class SymbolIdentityRegistry:
    """Validate ID/key claims across projects, shards, or a persisted index."""

    def __init__(self) -> None:
        self.keys_by_id: dict[int, str] = {}
        self.ids_by_key: dict[str, int] = {}
        self.sources_by_id: dict[int, str] = {}

    def register(
        self,
        symbol_key: str,
        *,
        symbol_id: int | None = None,
        source: str = "",
    ) -> int:
        if not isinstance(symbol_key, str) or not symbol_key:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_key must be a non-empty string"
            )
        claimed_id = compute_symbol_id(symbol_key) if symbol_id is None else int(symbol_id)
        if not 0 < claimed_id <= SYMBOL_ID_MAX:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_id must be in [1, {SYMBOL_ID_MAX}], "
                f"got {claimed_id}"
            )

        existing_key = self.keys_by_id.get(claimed_id)
        if existing_key is not None and existing_key != symbol_key:
            first_source = self.sources_by_id.get(claimed_id, "unknown source")
            raise SymbolIdentityError(
                "canonical symbol ID collision: "
                f"id={claimed_id} first={existing_key!r} ({first_source}) "
                f"second={symbol_key!r} ({source or 'unknown source'}). "
                "Refusing to emit aliased semantic supervision."
            )

        existing_id = self.ids_by_key.get(symbol_key)
        if existing_id is not None and existing_id != claimed_id:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: canonical key {symbol_key!r} was already "
                f"registered as ID {existing_id}, not {claimed_id}"
            )

        expected_id = compute_symbol_id(symbol_key)
        if claimed_id != expected_id:
            raise SymbolIdentityError(
                f"{source or 'symbol identity'}: symbol_id {claimed_id} does not match "
                f"the v{SYMBOL_IDENTITY_SCHEMA_VERSION} canonical ID {expected_id} for "
                f"{symbol_key!r}"
            )

        self.keys_by_id[claimed_id] = symbol_key
        self.ids_by_key[symbol_key] = claimed_id
        self.sources_by_id.setdefault(claimed_id, source or "unknown source")
        return claimed_id

    def register_records(
        self,
        records: object,
        *,
        source: str,
    ) -> list[dict[str, object]]:
        if records is None:
            records = []
        if not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
            raise SymbolIdentityError(
                f"{source}: {SYMBOL_IDENTITIES_COLUMN} must be a list of ID/key records"
            )
        normalized: list[dict[str, object]] = []
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise SymbolIdentityError(
                    f"{source}: {SYMBOL_IDENTITIES_COLUMN}[{index}] must be an object"
                )
            symbol_key = record.get("symbol_key")
            symbol_id = record.get("symbol_id")
            if not isinstance(symbol_key, str) or symbol_id is None:
                raise SymbolIdentityError(
                    f"{source}: {SYMBOL_IDENTITIES_COLUMN}[{index}] requires "
                    "symbol_id and symbol_key"
                )
            registered_id = self.register(
                symbol_key,
                symbol_id=int(symbol_id),
                source=source,
            )
            normalized.append(
                {"symbol_id": registered_id, "symbol_key": symbol_key}
            )
        return normalized

    def require_ids(self, symbol_ids: Iterable[int], *, source: str) -> None:
        missing = sorted(
            {
                int(symbol_id)
                for symbol_id in symbol_ids
                if int(symbol_id) != 0 and int(symbol_id) not in self.keys_by_id
            }
        )
        if missing:
            preview = missing[:8]
            suffix = "..." if len(missing) > len(preview) else ""
            raise SymbolIdentityError(
                f"{source}: semantic symbol IDs are missing canonical identity claims: "
                f"{preview}{suffix}"
            )

    def records(self, symbol_ids: Iterable[int] | None = None) -> list[dict[str, object]]:
        if symbol_ids is None:
            selected = sorted(self.keys_by_id)
        else:
            selected = sorted({int(value) for value in symbol_ids if int(value) != 0})
            self.require_ids(selected, source="symbol identity record export")
        return [
            {"symbol_id": symbol_id, "symbol_key": self.keys_by_id[symbol_id]}
            for symbol_id in selected
        ]


__all__ = [
    "EXTERNAL_PROVIDER_FILE_PREFIX",
    "EXTERNAL_PROVIDER_PROJECTS",
    "REPO_FILE_LOCATION_IDENTITY_PREFIX",
    "SYMBOL_IDENTITIES_COLUMN",
    "SYMBOL_IDENTITY_SCHEMA_METADATA_KEY",
    "SYMBOL_IDENTITY_SCHEMA_VERSION",
    "SYMBOL_ID_MAX",
    "ResolvedProjectIdentity",
    "RepoFileLocationIdentity",
    "UsrIdentity",
    "SymbolIdentityError",
    "SymbolIdentityRegistry",
    "canonical_external_provider_file",
    "canonical_external_usr_identity",
    "canonical_usr_identity",
    "compute_symbol_id",
    "external_provider_project",
    "is_repo_file_location_identity",
    "is_scoped_usr_identity",
    "parse_external_provider_file",
    "parse_repo_file_location_identity",
    "parse_usr_identity",
    "require_project_identity",
    "resolve_remote_project_identity",
]
