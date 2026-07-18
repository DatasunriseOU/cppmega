from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_root_v1_contract_assigns_ksh_and_python_without_renumbering() -> None:
    domain = _read(ROOT / "data" / "domain_schema_v1.json")
    tokenizer = _read(ROOT / "data" / "tokenizer_v2" / "tokenizer_contract_v1.json")

    assert domain["domain_kinds"]["KSH"] == 24
    assert domain["domain_kinds"]["PYTHON"] == 31
    assert domain["delimiter_roles"]["KSH"] == {
        "domain_id": 24,
        "start": "KSH_START",
        "end": "KSH_END",
    }
    assert domain["delimiter_roles"]["PYTHON"] == {
        "domain_id": 31,
        "start": "PYTHON_START",
        "end": "PYTHON_END",
    }
    assignments = tokenizer["reserved_role_assignments"]
    assert assignments["KSH_START"] == 245
    assert assignments["KSH_END"] == 246
    assert assignments["PYTHON_START"] == 247
    assert assignments["PYTHON_END"] == 248


def test_root_and_mlx_contract_json_semantics_match() -> None:
    mlx_root = ROOT.parent / "cppmega.mlx"
    root_domain = _read(ROOT / "data" / "domain_schema_v1.json")
    root_tokenizer = _read(
        ROOT / "data" / "tokenizer_v2" / "tokenizer_contract_v1.json"
    )
    mlx_domain = _read(mlx_root / "cppmega_mlx" / "data" / "domain_schema_v1.json")
    mlx_tokenizer = _read(
        mlx_root / "cppmega_mlx" / "tokenizer" / "tokenizer_contract_v1.json"
    )
    root_package_domain = _read(ROOT / "cppmega" / "data" / "domain_schema_v1.json")
    root_package_tokenizer = _read(
        ROOT / "cppmega" / "tokenizer" / "tokenizer_contract_v1.json"
    )

    assert root_domain == mlx_domain
    assert root_tokenizer == mlx_tokenizer
    assert root_package_domain == root_domain
    assert root_package_tokenizer == root_tokenizer


def test_case5_reader_accepts_only_known_complete_legacy_contract_triples() -> None:
    from cppmega.megatron.domain_route_contract import (
        is_accepted_case5_contract_hash_triple,
    )

    current = (
        "09fe81e915ee713004a1148abe54fbca2cf9ccfa9445901299a395d2b9fe253b",
        hashlib.sha256(
            (ROOT / "data" / "domain_schema_v1.json").read_bytes()
        ).hexdigest(),
        hashlib.sha256(
            (ROOT / "data" / "tokenizer_v2" / "tokenizer_contract_v1.json").read_bytes()
        ).hexdigest(),
    )
    legacy_case5 = (
        "1f2e35d7917409fc03704d32c2d55d0fb3e29f1bd9e60acca775a392cf2f53e6",
        "9c3517b5a3fda01c4f55d55bc0d12dff4af3edb3db6321bda6c22489061b4fdd",
        "c3bb669015c48e2049e3b82ccb8c98c6eceae0644f7da0b5b8600c573d7087a5",
    )

    assert is_accepted_case5_contract_hash_triple(*current)
    assert is_accepted_case5_contract_hash_triple(*legacy_case5)
    assert not is_accepted_case5_contract_hash_triple(
        legacy_case5[0], current[1], legacy_case5[2]
    )


def test_case5_reader_rejects_unknown_legacy_contract() -> None:
    from cppmega.megatron.domain_route_contract import (
        is_accepted_case5_contract_hash_triple,
    )

    assert not is_accepted_case5_contract_hash_triple("0" * 64, "1" * 64, "2" * 64)
