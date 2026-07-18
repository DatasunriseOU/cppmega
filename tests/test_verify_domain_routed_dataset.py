from __future__ import annotations

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from cppmega.data.domain_schema import DomainKind, DomainRoleKind, ParseConfidence
from cppmega.data.tokenizer_contract import DOMAIN_DELIMITER_TOKEN_IDS


def _load_verifier():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "verify_domain_routed_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("verify_domain_routed_dataset", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_verify_domain_routed_dataset_reports_domain_counts(tmp_path):
    mod = _load_verifier()
    root = tmp_path / "code" / "8"
    root.mkdir(parents=True)
    start = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_START"]
    end = DOMAIN_DELIMITER_TOKEN_IDS["COMPILER_ERROR_END"]
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "input_ids": [start, 10, 11, end, 0, 0, 0, 0],
                    "valid_token_count": 4,
                    "token_domain_ids": [int(DomainKind.COMPILER_ERROR)] * 4 + [0, 0, 0, 0],
                    "token_role_ids": [int(DomainRoleKind.DELIMITER), 32, 31, int(DomainRoleKind.DELIMITER), 0, 0, 0, 0],
                    "token_confidence_ids": [int(ParseConfidence.RAW)] * 4 + [0, 0, 0, 0],
                    "token_diagnostic_edges": [],
                    "build_kind": None,
                }
            ]
        ),
        root / "diag.parquet",
    )

    result = mod.verify_file("code", root / "diag.parquet", "8")
    report = mod.rollup([result], min_cpp_graph_coverage=0.0)

    assert report["valid_tokens"] == 4
    assert str(int(DomainKind.COMPILER_ERROR)) in report["domain_token_counts"]
    assert report["errors"] == []
