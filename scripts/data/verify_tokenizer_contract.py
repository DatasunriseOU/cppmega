#!/usr/bin/env python3
"""Verify the canonical tokenizer contract against the shipped tokenizer and every
repo that hard-codes vocab/special-token constants. Fail-closed (RULE #1): any real
mismatch raises with WHERE + WHAT and exits non-zero.

Single source of truth: ``cppmega_mlx/tokenizer/tokenizer_contract_v1.json``.

Checks
------
1. Contract <-> shipped ``tokenizer.json``: model vocab == contract vocab_size; every
   special-token id resolves to a token, and every reserved-role id resolves to its
   ``<RESERVED_N>`` slot (id == N).
2. Domain-delimiter reserved roles are complete START/END pairs and resolve to
   existing ``<RESERVED_N>`` slots.
3. Contract <-> mlx ``data/tokenizer_contract.py`` (REQUIRED/TOOL_USE special ids).
4. Contract <-> mlx ``tokenizer/cpp_tokenizer.py`` EXPECTED_VOCAB_SIZE.
5. Contract <-> nanochat ``scripts/tok_train_cpp.py`` --vocab_size default (if present).
6. NAM56R vocab dual-track: report model_factory NAM56R_FULL vs config default; assert
   the first-run pairing (model vocab paired with the 65536 artifact) is internally
   consistent. The 65536/131072 split itself is intentional and NOT a failure.

Usage:
    python verify_tokenizer_contract.py            # autodetect sibling repos
    python verify_tokenizer_contract.py --root /Volumes/external/sources
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class ContractError(Exception):
    """Fail-closed contract violation (WHERE + WHAT)."""


DOMAIN_DELIMITER_ROLE_PAIRS: tuple[str, ...] = (
    "CPP_CODE",
    "MAKE",
    "CMAKE",
    "NINJA",
    "BAZEL",
    "BASH",
    "ZSH",
    "SH",
    "TCSH",
    "COMPILER_DIAGNOSTIC",
    "BUILD_DIAGNOSTIC",
    "COMPILER_ERROR",
    "BUILD_ERROR",
    "LINKER_ERROR",
    "TEST_OUTPUT",
    "TOOL_OUTPUT",
)


def _read_json(p: Path) -> dict:
    if not p.is_file():
        raise ContractError(f"WHERE={p} WHAT=file not found")
    return json.loads(p.read_text())


def _grep_int(path: Path, pattern: str) -> list[int]:
    """Return every integer captured by ``pattern`` (one capture group) in ``path``."""
    if not path.is_file():
        return []
    out: list[int] = []
    rx = re.compile(pattern)
    for line in path.read_text().splitlines():
        m = rx.search(line)
        if m:
            out.append(int(m.group(1).replace("_", "")))
    return out


def check_contract_vs_tokenizer(contract: dict, tok_json: Path) -> list[str]:
    notes: list[str] = []
    tk = _read_json(tok_json)
    vocab = tk.get("model", {}).get("vocab", {})
    added = tk.get("added_tokens", [])
    id_to_tok = {a["id"]: a["content"] for a in added}

    want_vocab = contract["vocab_size"]
    if len(vocab) != want_vocab:
        raise ContractError(
            f"WHERE={tok_json} WHAT=model vocab {len(vocab)} != contract vocab_size {want_vocab}"
        )
    max_id = max([a["id"] for a in added], default=-1)
    if max_id >= want_vocab:
        raise ContractError(
            f"WHERE={tok_json} WHAT=added_token id {max_id} >= vocab_size {want_vocab}"
        )
    notes.append(f"tokenizer.json: vocab={len(vocab)} max_added_id={max_id} OK")

    # every special-token id must resolve to *some* added token
    for name, tid in contract["special_tokens"].items():
        if tid not in id_to_tok:
            raise ContractError(
                f"WHERE={tok_json} WHAT=special token {name}={tid} has no added_token entry"
            )

    # reserved-role ids must resolve to the matching <RESERVED_N> slot
    roles = {k: v for k, v in contract["reserved_role_assignments"].items() if not k.startswith("_")}
    for role, tid in roles.items():
        tok = id_to_tok.get(tid)
        if tok != f"<RESERVED_{tid}>":
            raise ContractError(
                f"WHERE={tok_json} WHAT=reserved role {role} id {tid} maps to {tok!r}, "
                f"expected '<RESERVED_{tid}>'"
            )
    notes.append(f"reserved roles bound to <RESERVED_N> slots: {roles} OK")
    notes += check_domain_delimiter_roles(contract, id_to_tok)
    return notes


def check_domain_delimiter_roles(contract: dict, id_to_tok: dict[int, str]) -> list[str]:
    assignments = contract.get("reserved_role_assignments")
    if not isinstance(assignments, dict):
        raise ContractError(
            "WHERE=contract WHAT=reserved_role_assignments must be an object"
        )

    domain_ids: dict[int, str] = {}
    for base in DOMAIN_DELIMITER_ROLE_PAIRS:
        for edge in ("START", "END"):
            role = f"{base}_{edge}"
            if role not in assignments:
                raise ContractError(
                    f"WHERE=contract WHAT=missing domain delimiter role {role}"
                )
            token_id = assignments[role]
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ContractError(
                    f"WHERE=contract WHAT=domain delimiter role {role} id must be int, "
                    f"got {token_id!r}"
                )
            token = id_to_tok.get(token_id)
            if token != f"<RESERVED_{token_id}>":
                raise ContractError(
                    f"WHERE=contract WHAT=domain delimiter role {role} id {token_id} "
                    f"maps to {token!r}, expected '<RESERVED_{token_id}>'"
                )
            previous = domain_ids.setdefault(token_id, role)
            if previous != role:
                raise ContractError(
                    f"WHERE=contract WHAT=domain delimiter id collision: id {token_id} "
                    f"maps to both {previous} and {role}"
                )

    return [
        "domain delimiter roles: "
        f"{len(DOMAIN_DELIMITER_ROLE_PAIRS)} START/END pairs in <RESERVED_N> slots OK"
    ]


def check_contract_vs_mlx_constants(contract: dict, mlx_root: Path) -> list[str]:
    notes: list[str] = []
    # tokenizer_contract.py REQUIRED/TOOL_USE ids must be a subset that agrees with us
    tc = mlx_root / "cppmega_mlx" / "data" / "tokenizer_contract.py"
    if tc.is_file():
        src = tc.read_text()
        special = contract["special_tokens"] | {"EOT": contract["aliases"]["EOT"]}
        for name, tid in re.findall(r'"([A-Z_]+)":\s*(\d+)', src):
            tid = int(tid)
            if name in special and special[name] != tid:
                raise ContractError(
                    f"WHERE={tc} WHAT={name}={tid} disagrees with contract {special[name]}"
                )
            if name in contract["reserved_role_assignments"]:
                want = contract["reserved_role_assignments"][name]
                if want != tid:
                    raise ContractError(
                        f"WHERE={tc} WHAT={name}={tid} disagrees with reserved role {want}"
                    )
        notes.append(f"{tc.name}: special/reserved ids agree with contract OK")

    # cpp_tokenizer.py EXPECTED_VOCAB_SIZE
    cpt = mlx_root / "cppmega_mlx" / "tokenizer" / "cpp_tokenizer.py"
    vs = _grep_int(cpt, r"EXPECTED_VOCAB_SIZE\s*=\s*([\d_]+)")
    if vs and vs[0] != contract["vocab_size"]:
        raise ContractError(
            f"WHERE={cpt} WHAT=EXPECTED_VOCAB_SIZE {vs[0]} != contract {contract['vocab_size']}"
        )
    if vs:
        notes.append(f"{cpt.name}: EXPECTED_VOCAB_SIZE={vs[0]} OK")
    return notes


def check_nanochat(contract: dict, nano_root: Path) -> list[str]:
    notes: list[str] = []
    tok_train = nano_root / "scripts" / "tok_train_cpp.py"
    vs = _grep_int(tok_train, r"--vocab_size.*default[=\s]+([\d_]+)")
    if not vs:
        vs = _grep_int(tok_train, r'default\s*=\s*([\d_]+).*vocab')
    if vs and vs[0] != contract["vocab_size"]:
        raise ContractError(
            f"WHERE={tok_train} WHAT=tok_train default vocab {vs[0]} != contract {contract['vocab_size']}"
        )
    if vs:
        notes.append(f"nanochat tok_train_cpp.py: default vocab={vs[0]} OK")
    elif tok_train.is_file():
        notes.append("nanochat tok_train_cpp.py: vocab default not auto-detected (skipped)")
    return notes


def check_nam56r_dual_track(contract: dict, mlx_root: Path) -> list[str]:
    """The 65536/131072 split is intentional; we only assert the first-run pairing
    is one of the two preserved contracts and report the divergence point."""
    notes: list[str] = []
    factory = mlx_root / "cppmega_mlx" / "recipes" / "model_factory.py"
    cfg = mlx_root / "cppmega_mlx" / "config" / "model.py"
    fv = _grep_int(factory, r"NAM56R_FULL_VOCAB_SIZE\s*=\s*([\d_]+)")
    cv = _grep_int(cfg, r"MEGACPP_TOKENIZER_VOCAB_SIZE\s*=\s*([\d_]+)")
    preserved = {
        contract["vocab_dual_track_note"]["local_profile_vocab_size"],
        contract["vocab_dual_track_note"]["megacpp_tokenizer_vocab_size"],
    }
    factory_v = fv[0] if fv else None
    config_v = cv[0] if cv else None
    if factory_v is not None and factory_v not in preserved:
        raise ContractError(
            f"WHERE={factory} WHAT=NAM56R_FULL_VOCAB_SIZE {factory_v} not a preserved contract {preserved}"
        )
    first = contract["vocab_dual_track_note"]["first_run_vocab_size"]
    if first not in preserved:
        raise ContractError(
            f"WHERE=contract WHAT=first_run_vocab_size {first} not in preserved {preserved}"
        )
    notes.append(
        f"NAM56R dual-track: factory NAM56R_FULL={factory_v} config MEGACPP={config_v} "
        f"first_run={first} (intentional split; first run pairs the {first} artifact)"
    )
    if factory_v is not None and factory_v != first:
        notes.append(
            f"  NOTE: factory NAM56R_FULL={factory_v} differs from first_run={first}; "
            f"ensure the model vocab and tokenizer artifact match at training launch."
        )
    return notes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    here = Path(__file__).resolve()
    default_root = here.parents[3]  # .../sources  (cppmega/scripts/data/<file>)
    ap.add_argument("--root", default=str(default_root), help="dir holding sibling repos")
    ap.add_argument("--contract", default="", help="path to tokenizer_contract_v1.json")
    args = ap.parse_args()

    root = Path(args.root)
    mlx_root = root / "cppmega.mlx"
    nano_root = root / "nanochat"
    contract_path = Path(args.contract) if args.contract else (
        mlx_root / "cppmega_mlx" / "tokenizer" / "tokenizer_contract_v1.json"
    )
    tok_json = mlx_root / "cppmega_mlx" / "tokenizer" / "tokenizer.json"

    contract = _read_json(contract_path)
    notes: list[str] = [f"contract: {contract_path} (v{contract['contract_version']})"]
    try:
        notes += check_contract_vs_tokenizer(contract, tok_json)
        notes += check_contract_vs_mlx_constants(contract, mlx_root)
        notes += check_nanochat(contract, nano_root)
        notes += check_nam56r_dual_track(contract, mlx_root)
    except ContractError as exc:
        print("TOKENIZER CONTRACT: FAIL")
        print(f"  {exc}")
        return 1

    print("TOKENIZER CONTRACT: OK")
    for n in notes:
        print(f"  - {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
