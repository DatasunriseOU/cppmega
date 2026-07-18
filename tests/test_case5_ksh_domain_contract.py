from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from cppmega.megatron.domain_route_contract import (
    DOMAIN_DELIMITER_ID_TO_DOMAIN,
    DOMAIN_DELIMITER_TOKEN_IDS,
    DOMAIN_EDGE_KINDS_BY_COLUMN,
    DOMAIN_END_DELIMITER_IDS,
    DOMAIN_START_DELIMITER_IDS,
    VALID_DOMAIN_IDS,
)


def test_case5_ksh_delimiter_and_cross_domain_routes_are_frozen() -> None:
    assert 24 in VALID_DOMAIN_IDS
    assert DOMAIN_DELIMITER_TOKEN_IDS["KSH_START"] == 245
    assert DOMAIN_DELIMITER_TOKEN_IDS["KSH_END"] == 246
    assert DOMAIN_DELIMITER_ID_TO_DOMAIN[245] == 24
    assert DOMAIN_DELIMITER_ID_TO_DOMAIN[246] == 24
    assert 245 in DOMAIN_START_DELIMITER_IDS
    assert 246 in DOMAIN_END_DELIMITER_IDS
    assert DOMAIN_EDGE_KINDS_BY_COLUMN["token_shell_edges"] == frozenset(
        range(40, 45)
    )
    assert DOMAIN_EDGE_KINDS_BY_COLUMN["token_cross_domain_edges"] == frozenset(
        {100}
    )


def test_megatron_ingress_accepts_frozen_ksh_prompt_sidecars() -> None:
    script = Path(__file__).resolve().parents[1] / "scripts" / "data_prep_parquet_to_megatron.py"
    spec = importlib.util.spec_from_file_location("case5_megatron_converter", script)
    assert spec is not None and spec.loader is not None
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)

    token_ids = [2, 245, 501, 502, 246, 215, 601, 602, 216, 3]
    values = {
        "token_domain_ids": [0, 24, 24, 24, 24, 43, 43, 43, 43, 0],
        "token_role_ids": [0, 1, 6, 0, 1, 1, 0, 0, 1, 0],
        "token_entity_ids": [0] * 10,
        "token_scope_ids": [0] * 10,
        "token_source_doc_ids": [1] * 10,
        "token_source_identity_ids": [99] * 10,
        "token_confidence_ids": [2, 4, 2, 2, 4, 4, 2, 2, 4, 2],
    }

    converter._validate_domain_route_sidecars(
        token_ids,
        values,
        shard_path="case5-ksh-fixture",
        row_idx=0,
    )

    shell = converter._normalize_edge_triples(
        [{"from": 2, "to": 3, "kind": 40}],
        column="token_shell_edges",
        shard_path="case5-ksh-fixture",
        row_idx=0,
    )
    diagnostic = converter._normalize_edge_triples(
        [{"from": 6, "to": 7, "kind": 64}],
        column="token_diagnostic_edges",
        shard_path="case5-ksh-fixture",
        row_idx=0,
    )
    cross_domain = converter._normalize_edge_triples(
        [{"from": 3, "to": 6, "kind": 100}],
        column="token_cross_domain_edges",
        shard_path="case5-ksh-fixture",
        row_idx=0,
    )

    np.testing.assert_array_equal(shell, np.array([[2, 3, 40]], dtype=np.int32))
    np.testing.assert_array_equal(diagnostic, np.array([[6, 7, 64]], dtype=np.int32))
    np.testing.assert_array_equal(cross_domain, np.array([[3, 6, 100]], dtype=np.int32))
