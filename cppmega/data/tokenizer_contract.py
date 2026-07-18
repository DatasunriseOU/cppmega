"""Dependency-free, fail-closed validation for the frozen tokenizer contract."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import re
from typing import Any


EXPECTED_CONTRACT_VERSION = 1
EXPECTED_VOCAB_SIZE = 65_536
EXPECTED_TOKEN_ROLE_CONTRACT_SHA256 = (
    "4775bc5dc3b90b4bd9d0949fdb54ac418c97c954e23b4b1724bb3b7fd596ac50"
)

REQUIRED_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "BOS": 2,
    "EOT": 3,
    "FIM_PREFIX": 4,
    "FIM_MIDDLE": 5,
    "FIM_SUFFIX": 6,
    "CODE_START": 7,
    "FIM_INSTRUCTION": 45,
    "SPACE": 46,
    "NL": 47,
}

TOOL_USE_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "BOS": 2,
    "EOS": 3,
    "CODE_START": 7,
    "CODE_END": 8,
    "THINK_START": 9,
    "THINK_END": 10,
    "QUERY_TOOL": 11,
    "TOOL_RESULT": 19,
}

EXPECTED_OBJECTIVE_BOUNDARY_ROLES = (
    "FILE_SEP",
    "DIFF_START",
    "DIFF_END",
    "COMMENT_START",
    "COMMENT_END",
    "SYMBOL_REF",
    "TYPE_INFO",
    "OVERLOAD_SET",
)

_TOKENIZER_DIR = Path(__file__).resolve().parents[1] / "tokenizer"
TOKENIZER_CONTRACT_PATH = _TOKENIZER_DIR / "tokenizer_contract_v1.json"
_ROLE_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")
_RESERVED_TOKEN_RE = re.compile(r"^<RESERVED_(\d+)>$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _DuplicateJsonKeyError(ValueError):
    pass


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _decode_json_object(raw: bytes, *, where: Path, kind: str) -> dict[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_json_keys
        )
    except (UnicodeDecodeError, json.JSONDecodeError, _DuplicateJsonKeyError) as exc:
        raise ValueError(f"cannot load {kind} {where}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{kind} {where} must be an object")
    return payload


def _read_contract_bytes(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot load tokenizer contract {path}: {exc}") from exc
    try:
        contract = _decode_json_object(raw, where=path, kind="tokenizer contract")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    return contract, raw


def _read_contract(path: Path) -> dict[str, Any]:
    contract, _raw = _read_contract_bytes(path)
    return contract


def _read_json_object(path: Path, *, kind: str) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"cannot load {kind} {path}: {exc}") from exc
    return _decode_json_object(raw, where=path, kind=kind)


def canonical_json_sha256(value: object) -> str:
    """Hash JSON semantics independently of whitespace and object key order."""

    canonical = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _require_mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _require_int(value: object, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{name} must be int")
    return value


def _read_role_id_mapping(
    contract: Mapping[str, object],
    key: str,
    *,
    vocab_size: int,
    allow_doc: bool = False,
) -> dict[str, int]:
    raw_mapping = _require_mapping(contract.get(key), name=key)
    result: dict[str, int] = {}
    for raw_role, raw_id in raw_mapping.items():
        if allow_doc and raw_role == "_doc":
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError(f"{key}._doc must be a non-empty string")
            continue
        if not isinstance(raw_role, str) or not _ROLE_NAME_RE.fullmatch(raw_role):
            raise ValueError(f"{key} contains invalid role name {raw_role!r}")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise ValueError(f"{key}.{raw_role} id must be int")
        if not 0 <= raw_id < vocab_size:
            raise ValueError(
                f"{key}.{raw_role} id {raw_id} is outside vocab size {vocab_size}"
            )
        result[raw_role] = raw_id
    if not result:
        raise ValueError(f"{key} must define at least one role")
    return result


def _reject_id_collisions(mapping: Mapping[str, int], *, name: str) -> None:
    seen: dict[int, str] = {}
    for role, token_id in mapping.items():
        existing = seen.setdefault(token_id, role)
        if existing != role:
            raise ValueError(
                f"{name} id collision: id {token_id} maps to both {existing} and {role}"
            )


def _validate_sequence_contracts(
    contract: Mapping[str, object],
    *,
    named_ids: Mapping[str, int],
) -> None:
    sequences = _require_mapping(
        contract.get("sequence_contracts"), name="sequence_contracts"
    )
    expected_references = {
        "line_comment_eol": {"terminator": "NL", "status": "active_v1"},
        "comment_objective": {
            "opening": "COMMENT_START",
            "closing": "COMMENT_END",
            "status": "active_v1",
        },
        "docstring": {
            "representation": "domain_role_sidecar",
            "status": "no_dedicated_v1_delimiter_pair",
        },
        "raw_control_literals": {
            "representation": "added_special_tokens",
            "policy": (
                "reject in untrusted raw text; trusted templates insert control "
                "ids explicitly"
            ),
            "status": "ambiguous_in_v1",
        },
        "fim": {
            "opening": "FIM_PREFIX",
            "middle": "FIM_MIDDLE",
            "suffix": "FIM_SUFFIX",
            "closing": "EOT",
            "status": "legacy_v1",
        },
        "ifim": {
            "opening": "FIM_INSTRUCTION",
            "fim_opening": "FIM_PREFIX",
            "middle": "FIM_MIDDLE",
            "suffix": "FIM_SUFFIX",
            "closing": "EOT",
            "status": "legacy_v1",
        },
    }
    for sequence_name, expected in expected_references.items():
        sequence = _require_mapping(
            sequences.get(sequence_name),
            name=f"sequence_contracts.{sequence_name}",
        )
        for field, expected_value in expected.items():
            actual = sequence.get(field)
            if actual != expected_value:
                raise ValueError(
                    f"sequence_contracts.{sequence_name}.{field} must be "
                    f"{expected_value!r}, got {actual!r}"
                )
            if (
                field
                in {
                    "terminator",
                    "opening",
                    "closing",
                    "middle",
                    "suffix",
                    "fim_opening",
                }
                and expected_value not in named_ids
            ):
                raise ValueError(
                    f"sequence_contracts.{sequence_name}.{field} references "
                    f"unknown token role {expected_value!r}"
                )
    comment = _require_mapping(
        sequences["comment_objective"],
        name="sequence_contracts.comment_objective",
    )
    if named_ids[str(comment["opening"])] == named_ids[str(comment["closing"])]:
        raise ValueError("comment_objective opening/closing ids collide")
    if named_ids["NL"] != 47:
        raise ValueError("line-comment EOL must remain the frozen NL id 47")


def _validate_migration_plan(
    contract: Mapping[str, object], *, vocab_size: int
) -> None:
    plan = _require_mapping(contract.get("migration_plan"), name="migration_plan")
    expected = {
        "target_contract_version": 2,
        "status": "blocked_until_retokenization_and_checkpoint_training",
        "frozen_v1_artifact_mutation": "forbidden",
        "vocab_size": vocab_size,
    }
    for key, expected_value in expected.items():
        actual = plan.get(key)
        if actual != expected_value:
            raise ValueError(
                f"migration_plan.{key} must be {expected_value!r}, got {actual!r}"
            )
    steps = plan.get("required_steps")
    if (
        not isinstance(steps, list)
        or len(steps) < 4
        or any(not isinstance(step, str) or not step.strip() for step in steps)
    ):
        raise ValueError("migration_plan.required_steps must contain explicit steps")


def _validated_contract_sections(
    contract: Mapping[str, object],
) -> tuple[dict[str, int], dict[str, int], tuple[str, ...], dict[str, int]]:
    version = contract.get("contract_version")
    if version != EXPECTED_CONTRACT_VERSION:
        raise ValueError(
            f"contract_version must be {EXPECTED_CONTRACT_VERSION}, got {version!r}"
        )
    vocab_size = contract.get("vocab_size")
    if (
        not isinstance(vocab_size, int)
        or isinstance(vocab_size, bool)
        or vocab_size != EXPECTED_VOCAB_SIZE
    ):
        raise ValueError(
            f"vocab_size must be frozen at {EXPECTED_VOCAB_SIZE}, got {vocab_size!r}"
        )

    special = _read_role_id_mapping(contract, "special_tokens", vocab_size=vocab_size)
    aliases = _read_role_id_mapping(contract, "aliases", vocab_size=vocab_size)
    reserved = _read_role_id_mapping(
        contract,
        "reserved_role_assignments",
        vocab_size=vocab_size,
        allow_doc=True,
    )
    _reject_id_collisions(special, name="special_tokens")
    _reject_id_collisions(reserved, name="reserved_role_assignments")

    duplicate_roles = (set(special) & set(aliases)) | (set(special) & set(reserved))
    duplicate_roles |= set(aliases) & set(reserved)
    if duplicate_roles:
        raise ValueError(
            f"token roles appear in multiple sections: {sorted(duplicate_roles)}"
        )
    collisions = set(special.values()) & set(reserved.values())
    if collisions:
        raise ValueError(
            "reserved roles collide with canonical special-token ids: "
            f"{sorted(collisions)}"
        )
    starts = {
        role.removesuffix("_START") for role in reserved if role.endswith("_START")
    }
    ends = {role.removesuffix("_END") for role in reserved if role.endswith("_END")}
    if starts != ends:
        missing = sorted(
            [f"{base}_END" for base in starts - ends]
            + [f"{base}_START" for base in ends - starts]
        )
        raise ValueError(f"unpaired domain delimiter roles: {missing}")
    canonical_ids = set(special.values())
    for alias, token_id in aliases.items():
        if token_id not in canonical_ids:
            raise ValueError(
                f"alias {alias} id {token_id} does not reference a special token"
            )

    named_ids = {**special, **aliases}
    for role, expected_id in {
        **REQUIRED_SPECIAL_TOKEN_IDS,
        **TOOL_USE_SPECIAL_TOKEN_IDS,
    }.items():
        actual_id = named_ids.get(role)
        if actual_id != expected_id:
            raise ValueError(
                f"contract token role {role} must use id {expected_id}, got {actual_id}"
            )

    raw_objective_roles = contract.get("objective_boundary_roles")
    if not isinstance(raw_objective_roles, list) or any(
        not isinstance(role, str) for role in raw_objective_roles
    ):
        raise ValueError("objective_boundary_roles must be a list of token roles")
    objective_roles = tuple(raw_objective_roles)
    if objective_roles != EXPECTED_OBJECTIVE_BOUNDARY_ROLES:
        raise ValueError(
            "objective_boundary_roles drift: "
            f"expected {EXPECTED_OBJECTIVE_BOUNDARY_ROLES}, got {objective_roles}"
        )
    if len(objective_roles) != len(set(objective_roles)):
        raise ValueError("objective_boundary_roles contains duplicates")
    missing_objective_roles = sorted(set(objective_roles) - set(special))
    if missing_objective_roles:
        raise ValueError(
            f"objective boundary roles are not special tokens: {missing_objective_roles}"
        )

    artifact = _require_mapping(
        contract.get("artifact_contract"), name="artifact_contract"
    )
    digest = artifact.get("canonical_json_sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError(
            "artifact_contract.canonical_json_sha256 must be lowercase SHA-256"
        )
    if artifact.get("model_type") != "BPE":
        raise ValueError("artifact_contract.model_type must be 'BPE'")
    if artifact.get("model_unk_token") != "<UNK>":
        raise ValueError("artifact_contract.model_unk_token must be '<UNK>'")
    numeric_artifact_fields = {
        "vocab_id_min": 0,
        "vocab_id_max": vocab_size - 1,
        "added_tokens_count": 7200,
        "added_token_id_min": 0,
        "added_token_id_max": 7199,
        "reserved_token_count": 4896,
    }
    for field, expected_value in numeric_artifact_fields.items():
        actual = artifact.get(field)
        if actual != expected_value:
            raise ValueError(
                f"artifact_contract.{field} must be {expected_value}, got {actual!r}"
            )

    expected_role_digest = contract.get("token_role_contract_sha256")
    if expected_role_digest != EXPECTED_TOKEN_ROLE_CONTRACT_SHA256:
        raise ValueError(
            "token_role_contract_sha256 must remain frozen for contract v1: "
            f"expected {EXPECTED_TOKEN_ROLE_CONTRACT_SHA256}, "
            f"got {expected_role_digest!r}"
        )
    actual_role_digest = canonical_json_sha256(
        {
            "special_tokens": special,
            "aliases": aliases,
            "reserved_role_assignments": reserved,
        }
    )
    if expected_role_digest != actual_role_digest:
        raise ValueError(
            "token role assignment SHA-256 mismatch: "
            f"expected {expected_role_digest!r}, got {actual_role_digest}"
        )

    _validate_sequence_contracts(contract, named_ids=named_ids)
    _validate_migration_plan(contract, vocab_size=vocab_size)
    return special, aliases, objective_roles, reserved


def validate_tokenizer_contract(contract: Mapping[str, object]) -> None:
    """Validate every role, collision boundary, sequence, and migration gate."""

    _validated_contract_sections(contract)


def _derive_domain_delimiter_token_ids(
    contract: Mapping[str, object],
) -> dict[str, int]:
    _special, _aliases, _objective_roles, assignments = _validated_contract_sections(
        contract
    )
    starts = {
        role.removesuffix("_START") for role in assignments if role.endswith("_START")
    }
    ends = {role.removesuffix("_END") for role in assignments if role.endswith("_END")}
    if starts != ends:
        missing = sorted(
            [f"{base}_END" for base in starts - ends]
            + [f"{base}_START" for base in ends - starts]
        )
        raise ValueError(f"unpaired domain delimiter roles: {missing}")
    result = {
        f"{base}_{edge}": assignments[f"{base}_{edge}"]
        for base in sorted(starts)
        for edge in ("START", "END")
    }
    if not result:
        raise ValueError("tokenizer contract defines no domain delimiter pairs")
    for base in starts:
        if result[f"{base}_START"] == result[f"{base}_END"]:
            raise ValueError(f"{base} opening/closing delimiter ids collide")
    return result


TOKENIZER_CONTRACT, _TOKENIZER_CONTRACT_BYTES = _read_contract_bytes(
    TOKENIZER_CONTRACT_PATH
)
(
    SPECIAL_TOKEN_IDS,
    SPECIAL_TOKEN_ALIASES,
    _OBJECTIVE_BOUNDARY_ROLES,
    RESERVED_ROLE_TOKEN_IDS,
) = _validated_contract_sections(TOKENIZER_CONTRACT)
TOKENIZER_CONTRACT_SHA256 = hashlib.sha256(_TOKENIZER_CONTRACT_BYTES).hexdigest()
TOKENIZER_CONTRACT_SHA256_METADATA_KEY = "cppmega.tokenizer_contract_sha256"
OBJECTIVE_BOUNDARY_TOKEN_IDS = {
    role: SPECIAL_TOKEN_IDS[role] for role in _OBJECTIVE_BOUNDARY_ROLES
}
DOMAIN_DELIMITER_TOKEN_IDS = _derive_domain_delimiter_token_ids(TOKENIZER_CONTRACT)
DOMAIN_DELIMITER_CONTRACT_METADATA_KEY = "cppmega.domain_delimiter_contract_sha256"
DOMAIN_DELIMITER_CONTRACT_SHA256 = canonical_json_sha256(DOMAIN_DELIMITER_TOKEN_IDS)
TOKEN_ROLE_CONTRACT_SHA256 = canonical_json_sha256(
    {
        "special_tokens": SPECIAL_TOKEN_IDS,
        "aliases": SPECIAL_TOKEN_ALIASES,
        "reserved_role_assignments": RESERVED_ROLE_TOKEN_IDS,
    }
)
CONTROL_TOKEN_LITERAL_IDS = {
    **{f"<{role}>": token_id for role, token_id in SPECIAL_TOKEN_IDS.items()},
    **{
        f"<RESERVED_{token_id}>": token_id
        for token_id in RESERVED_ROLE_TOKEN_IDS.values()
    },
}

SpecialTokenMapping = Mapping[int, str] | Mapping[str, int]


def validate_required_special_token_ids(mapping: SpecialTokenMapping) -> None:
    """Validate required cppmega special-token ids in either mapping direction."""

    token_to_id = _normalize_special_token_mapping(mapping)
    for token, expected_id in REQUIRED_SPECIAL_TOKEN_IDS.items():
        if token not in token_to_id:
            raise ValueError(f"missing required special token {token!r}")
        actual_id = token_to_id[token]
        if actual_id != expected_id:
            raise ValueError(
                f"special token {token!r} must use id {expected_id}, got {actual_id}"
            )

    seen_ids: dict[int, str] = {}
    for token, token_id in token_to_id.items():
        existing = seen_ids.setdefault(token_id, token)
        if existing != token:
            raise ValueError(
                f"special token id collision: id {token_id} maps to both "
                f"{existing!r} and {token!r}"
            )


def validate_untrusted_tokenizer_text(
    text: str,
    *,
    where: str | Path = "text",
) -> None:
    """Reject raw text that would be interpreted as an active control token."""

    if not isinstance(text, str):
        raise TypeError(f"{where}: tokenizer text must be str")
    hits = (
        (index, literal, token_id)
        for literal, token_id in CONTROL_TOKEN_LITERAL_IDS.items()
        if (index := text.find(literal)) >= 0
    )
    hit = min(hits, default=None)
    if hit is None:
        return
    index, literal, token_id = hit
    raise ValueError(
        f"{where}: raw text contains tokenizer control literal {literal!r} "
        f"at character {index} (id {token_id}); trusted templates must insert "
        "control ids outside untrusted text"
    )


def _extract_artifact_vocab(
    tokenizer: Mapping[str, object], *, where: Path, vocab_size: int
) -> tuple[dict[str, int], dict[int, str], Mapping[str, object]]:
    model = _require_mapping(tokenizer.get("model"), name=f"{where}: model")
    raw_vocab = _require_mapping(model.get("vocab"), name=f"{where}: model.vocab")
    vocab: dict[str, int] = {}
    id_to_token: dict[int, str] = {}
    for raw_token, raw_id in raw_vocab.items():
        if not isinstance(raw_token, str):
            raise ValueError(f"{where}: model vocab tokens must be strings")
        if not isinstance(raw_id, int) or isinstance(raw_id, bool):
            raise ValueError(f"{where}: model vocab id for {raw_token!r} must be int")
        existing = id_to_token.setdefault(raw_id, raw_token)
        if existing != raw_token:
            raise ValueError(
                f"{where}: model vocab id {raw_id} maps to both "
                f"{existing!r} and {raw_token!r}"
            )
        vocab[raw_token] = raw_id
    if len(vocab) != vocab_size:
        raise ValueError(f"{where}: expected vocab size {vocab_size}, got {len(vocab)}")
    expected_ids = set(range(vocab_size))
    actual_ids = set(id_to_token)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:8]
        extra = sorted(actual_ids - expected_ids)[:8]
        raise ValueError(
            f"{where}: model vocab ids must be contiguous 0..{vocab_size - 1}; "
            f"missing={missing}, extra={extra}"
        )
    return vocab, id_to_token, model


def _extract_added_tokens(
    tokenizer: Mapping[str, object],
    *,
    where: Path,
    vocab: Mapping[str, int],
) -> tuple[dict[int, Mapping[str, object]], dict[str, int]]:
    raw_added = tokenizer.get("added_tokens")
    if not isinstance(raw_added, list):
        raise ValueError(f"{where}: added_tokens must be a list")
    by_id: dict[int, Mapping[str, object]] = {}
    by_content: dict[str, int] = {}
    for index, item in enumerate(raw_added):
        if not isinstance(item, Mapping):
            raise ValueError(f"{where}: added_tokens[{index}] must be an object")
        token_id = item.get("id")
        content = item.get("content")
        if not isinstance(token_id, int) or isinstance(token_id, bool):
            raise ValueError(f"{where}: added_tokens[{index}].id must be int")
        if not isinstance(content, str):
            raise ValueError(f"{where}: added_tokens[{index}].content must be str")
        if token_id in by_id:
            raise ValueError(f"{where}: duplicate added-token id {token_id}")
        if content in by_content:
            raise ValueError(f"{where}: duplicate added-token content {content!r}")
        if vocab.get(content) != token_id:
            raise ValueError(
                f"{where}: added token {content!r} id {token_id} disagrees "
                f"with model vocab id {vocab.get(content)!r}"
            )
        if item.get("special") is not True:
            raise ValueError(f"{where}: added token {content!r} must be special")
        for flag in ("single_word", "lstrip", "rstrip", "normalized"):
            if item.get(flag) is not False:
                raise ValueError(
                    f"{where}: added token {content!r} must set {flag}=false"
                )
        by_id[token_id] = item
        by_content[content] = token_id
    return by_id, by_content


def validate_tokenizer_payload_contract(
    tokenizer: Mapping[str, object],
    *,
    where: str | Path,
    contract: Mapping[str, object] | None = None,
    require_frozen_artifact: bool = False,
) -> dict[str, int]:
    """Validate a tokenizer payload against all active token-role assignments."""

    artifact_path = Path(where)
    active_contract = TOKENIZER_CONTRACT if contract is None else contract
    special, _aliases, _objective_roles, reserved = _validated_contract_sections(
        active_contract
    )
    vocab_size = _require_int(active_contract.get("vocab_size"), name="vocab_size")
    vocab, id_to_token, model = _extract_artifact_vocab(
        tokenizer, where=artifact_path, vocab_size=vocab_size
    )
    added_by_id, _added_by_content = _extract_added_tokens(
        tokenizer, where=artifact_path, vocab=vocab
    )

    artifact_contract = _require_mapping(
        active_contract.get("artifact_contract"), name="artifact_contract"
    )
    if model.get("type") != artifact_contract["model_type"]:
        raise ValueError(
            f"{artifact_path}: model type must be {artifact_contract['model_type']!r}, "
            f"got {model.get('type')!r}"
        )
    if model.get("unk_token") != artifact_contract["model_unk_token"]:
        raise ValueError(
            f"{artifact_path}: model unk_token must be "
            f"{artifact_contract['model_unk_token']!r}, got {model.get('unk_token')!r}"
        )

    expected_tokens = {
        **{token_id: f"<{role}>" for role, token_id in special.items()},
        **{token_id: f"<RESERVED_{token_id}>" for token_id in reserved.values()},
    }
    for token_id, expected_token in expected_tokens.items():
        actual_token = id_to_token[token_id]
        if actual_token != expected_token:
            raise ValueError(
                f"{artifact_path}: id {token_id} must map to {expected_token!r}, "
                f"got {actual_token!r}"
            )
        added = added_by_id.get(token_id)
        if added is None or added.get("content") != expected_token:
            raise ValueError(
                f"{artifact_path}: control token {expected_token!r} id {token_id} "
                "must be present in added_tokens"
            )

    if require_frozen_artifact:
        added_ids = set(added_by_id)
        expected_added_count = _require_int(
            artifact_contract.get("added_tokens_count"),
            name="artifact_contract.added_tokens_count",
        )
        if len(added_by_id) != expected_added_count:
            raise ValueError(
                f"{artifact_path}: expected {expected_added_count} added tokens, "
                f"got {len(added_by_id)}"
            )
        expected_added_ids = set(
            range(
                _require_int(
                    artifact_contract.get("added_token_id_min"),
                    name="artifact_contract.added_token_id_min",
                ),
                _require_int(
                    artifact_contract.get("added_token_id_max"),
                    name="artifact_contract.added_token_id_max",
                )
                + 1,
            )
        )
        if added_ids != expected_added_ids:
            raise ValueError(
                f"{artifact_path}: added-token ids must be contiguous "
                f"{min(expected_added_ids)}..{max(expected_added_ids)}"
            )
        reserved_count = sum(
            1
            for token, token_id in vocab.items()
            if (match := _RESERVED_TOKEN_RE.fullmatch(token))
            and int(match.group(1)) == token_id
        )
        expected_reserved_count = _require_int(
            artifact_contract.get("reserved_token_count"),
            name="artifact_contract.reserved_token_count",
        )
        if reserved_count != expected_reserved_count:
            raise ValueError(
                f"{artifact_path}: expected {artifact_contract['reserved_token_count']} "
                f"reserved slots, got {reserved_count}"
            )
        digest = canonical_json_sha256(tokenizer)
        expected_digest = str(artifact_contract["canonical_json_sha256"])
        if digest != expected_digest:
            raise ValueError(
                f"{artifact_path}: tokenizer semantic SHA-256 {digest} != "
                f"frozen {expected_digest}"
            )
    return _derive_domain_delimiter_token_ids(active_contract)


def validate_tokenizer_artifact_contract(
    tokenizer_path: str | Path,
    *,
    contract_path: str | Path | None = None,
    require_frozen_artifact: bool = False,
) -> dict[str, int]:
    """Load and validate one tokenizer artifact without importing tokenizers."""

    path = Path(tokenizer_path)
    contract = (
        TOKENIZER_CONTRACT
        if contract_path is None
        else _read_contract(Path(contract_path))
    )
    tokenizer = _read_json_object(path, kind="tokenizer artifact")
    return validate_tokenizer_payload_contract(
        tokenizer,
        where=path,
        contract=contract,
        require_frozen_artifact=require_frozen_artifact,
    )


def validate_huggingface_tokenizer_sidecars(tokenizer_dir: str | Path) -> None:
    """Validate the optional Hugging Face sidecars shipped by cppmega tokenizer_v2."""

    directory = Path(tokenizer_dir)
    special_map_path = directory / "special_tokens_map.json"
    config_path = directory / "tokenizer_config.json"
    special_map = _read_json_object(special_map_path, kind="special-token map")
    config = _read_json_object(config_path, kind="tokenizer config")
    expected = {
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
    }
    for field, token in expected.items():
        if special_map.get(field) != token:
            raise ValueError(
                f"{special_map_path}: {field} must be {token!r}, "
                f"got {special_map.get(field)!r}"
            )
        if config.get(field) != token:
            raise ValueError(
                f"{config_path}: {field} must be {token!r}, got {config.get(field)!r}"
            )
    decoder = _require_mapping(
        config.get("added_tokens_decoder"),
        name=f"{config_path}: added_tokens_decoder",
    )
    for role in ("PAD", "UNK", "BOS", "EOS"):
        token_id = SPECIAL_TOKEN_IDS[role]
        entry = _require_mapping(
            decoder.get(str(token_id)),
            name=f"{config_path}: added_tokens_decoder.{token_id}",
        )
        if entry.get("content") != f"<{role}>" or entry.get("special") is not True:
            raise ValueError(
                f"{config_path}: added_tokens_decoder.{token_id} must define "
                f"special token <{role}>"
            )


def validate_checked_out_tokenizer_contract(
    repo_root: str | Path | None = None,
) -> dict[str, int]:
    """Validate the contract and exact frozen tokenizer from this checkout."""

    tokenizer_dir = (
        Path(repo_root).resolve() / "cppmega" / "tokenizer"
        if repo_root is not None
        else _TOKENIZER_DIR
    )
    delimiter_ids = validate_tokenizer_artifact_contract(
        tokenizer_dir / "tokenizer.json",
        contract_path=tokenizer_dir / "tokenizer_contract_v1.json",
        require_frozen_artifact=True,
    )
    if tokenizer_dir == _TOKENIZER_DIR and delimiter_ids != DOMAIN_DELIMITER_TOKEN_IDS:
        raise ValueError(
            "imported domain delimiter mapping differs from checked-out contract"
        )
    return delimiter_ids


def _normalize_special_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    if not mapping:
        raise ValueError("special token mapping must not be empty")

    keys_are_ids = all(_is_int_key(key) for key in mapping)
    values_are_tokens = all(isinstance(value, str) for value in mapping.values())
    keys_are_tokens = all(isinstance(key, str) for key in mapping)
    values_are_ids = all(_is_int_key(value) for value in mapping.values())

    if keys_are_ids and values_are_tokens:
        return _invert_id_to_token_mapping(mapping)
    if keys_are_tokens and values_are_ids:
        return {str(token): int(token_id) for token, token_id in mapping.items()}

    raise ValueError(
        "special token mapping must be consistently id->token or token->id"
    )


def _invert_id_to_token_mapping(mapping: SpecialTokenMapping) -> dict[str, int]:
    token_to_id: dict[str, int] = {}
    for raw_id, raw_token in mapping.items():
        token_id = int(raw_id)
        token = str(raw_token)
        existing = token_to_id.setdefault(token, token_id)
        if existing != token_id:
            raise ValueError(
                f"special token collision: token {token!r} maps to both "
                f"{existing} and {token_id}"
            )
    return token_to_id


def _is_int_key(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


__all__ = [
    "CONTROL_TOKEN_LITERAL_IDS",
    "DOMAIN_DELIMITER_TOKEN_IDS",
    "OBJECTIVE_BOUNDARY_TOKEN_IDS",
    "DOMAIN_DELIMITER_CONTRACT_METADATA_KEY",
    "DOMAIN_DELIMITER_CONTRACT_SHA256",
    "EXPECTED_CONTRACT_VERSION",
    "EXPECTED_TOKEN_ROLE_CONTRACT_SHA256",
    "EXPECTED_VOCAB_SIZE",
    "REQUIRED_SPECIAL_TOKEN_IDS",
    "RESERVED_ROLE_TOKEN_IDS",
    "SPECIAL_TOKEN_ALIASES",
    "SPECIAL_TOKEN_IDS",
    "SpecialTokenMapping",
    "TOKENIZER_CONTRACT",
    "TOKENIZER_CONTRACT_PATH",
    "TOKENIZER_CONTRACT_SHA256",
    "TOKENIZER_CONTRACT_SHA256_METADATA_KEY",
    "TOKEN_ROLE_CONTRACT_SHA256",
    "TOOL_USE_SPECIAL_TOKEN_IDS",
    "canonical_json_sha256",
    "validate_checked_out_tokenizer_contract",
    "validate_huggingface_tokenizer_sidecars",
    "validate_required_special_token_ids",
    "validate_tokenizer_artifact_contract",
    "validate_tokenizer_contract",
    "validate_tokenizer_payload_contract",
    "validate_untrusted_tokenizer_text",
]
