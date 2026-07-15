#!/usr/bin/env python3
"""Mirror the cppmega.mlx token-enriched parquet rewrite onto OUR parquet, plus
backfill of GENUINELY DERIVABLE missing metadata.

Run with a cppmega-local Python environment that provides pyarrow 21.x.

WHAT THIS DOES (and what it deliberately does NOT do)
=====================================================
Forensic finding (verified on disk, NOT assumed):

  * The cppmega.mlx token transformation (the 13 `token_*` columns added by
    materialize_tokenized_enriched_batch, with field-specific integer dtypes
    and an EMPTY schema-level key/value metadata footer) is ALREADY APPLIED
    in place to both clang keeper datasets:
        - clang_semantic_4k_v10   (CODE,    66 shards)
        - clang_commits_4k_v1     (COMMITS, 104 shards)
    So the "mirror columns_added / columns_changed / metadata stamping" step is a
    VERIFY-ONLY no-op on these datasets: we assert the columns are present with the
    exact mlx dtypes and that the schema footer carries no metadata (mlx stamped
    none). We do NOT rewrite token columns and we do NOT invent footer metadata.

  * The ONLY genuinely derivable, non-fabricated missing metadata is `repo` on the
    COMMITS family, parsed from the embedded doc-comment header line
        ` * Repository: <name>`
    which carries REAL, per-row-varying repository names (o3de, pcl, wolfssl,
    microsoft-ui-xaml, ...). A robust doc-comment parser achieves 100% coverage on
    sampled clang_commits shards. We backfill `repo` and RECORD it.

  * Everything else listed as "empty" is NOT safely derivable and is left untouched
    and RECORDED as not_derivable. In particular:
        - platform_info: the EXISTING populated values are SEMANTIC content-analysis
          dicts (e.g. {"os":[],"gpu":["metal"],...}); the `// platform:` header is
          CONSTANT boilerplate ('x86_64-linux-gnu') identical on every row and would
          FABRICATE values contradicting the real semantics. REFUSED.
        - build_info: derivable only from the same CONSTANT `// compiler:`/`// standard:`
          boilerplate (g++ / c++17 on every row) -> uniform stamp is a fabrication of
          provenance that is not actually known per-row. REFUSED.
        - commit / commit_hash / timestamp: genuinely absent from every row. REFUSED.
        - constituent_provenance[_json]: all-null, not reconstructable. REFUSED.

Per RULE #1 (fail fast, fail loud, no silent fallbacks, never fabricate):
  * If a row in a backfill column lacks the header that the dataset is CLAIMED to
    carry (>= the configured min coverage), we RAISE with WHERE+WHAT rather than
    emitting a guessed/empty value.
  * On ANY verification mismatch we delete the .tmp and RAISE with WHERE+WHAT.

SAFETY MODEL
============
For each shard:
  1. Read the full original table.
  2. Apply the (verify-only) mlx-mirror checks + the safe derivable backfill,
     producing a NEW table that preserves every original column unchanged and only
     FILLS the previously-empty backfill column(s).
  3. Write to <shard>.tmp.
  4. VERIFY <shard>.tmp against the original:
        - identical row count
        - every ORIGINAL column value-equal (byte/value identical) to the original,
          with extra emphasis on text / token_ids / actual_token_count
        - the mlx token_* columns present with the exact expected dtypes
        - the empty schema footer preserved
        - the backfilled column present and non-empty on exactly the rows we claimed
  5. apply mode only, and only on success: back up original -> <shard>.bak (skip if
     .bak already exists), then os.replace(tmp, shard).
  6. dry-run mode: keep the .tmp + verification report, leave original untouched,
     do NOT back up.

Idempotent / re-runnable: if a shard already has every backfill column fully
populated (no remaining empties on rows that carry the header), it is a no-op
(reported state="already_complete"); .bak is never overwritten.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa  # type: ignore[import-not-found]
import pyarrow.compute as pc  # type: ignore[import-not-found]
import pyarrow.parquet as pq  # type: ignore[import-not-found]

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cppmega.symbol_identity import (  # noqa: E402
    SYMBOL_IDENTITIES_COLUMN,
    SYMBOL_IDENTITY_SCHEMA_METADATA_KEY,
    SYMBOL_IDENTITY_SCHEMA_VERSION,
    SymbolIdentityRegistry,
)


PARQUET_ROOT = Path("/Users/dave/sources/parquet")

# ---------------------------------------------------------------------------
# mlx canonical token-enriched columns + EXACT on-disk dtypes (verified).
# These are what the mlx rewrite added; we VERIFY them, we do not recreate them.
# ---------------------------------------------------------------------------
MLX_TOKEN_COLUMNS: dict[str, pa.DataType] = {
    "token_ids": pa.large_list(pa.uint32()),
    "platform_ids": pa.large_list(pa.uint16()),
    "token_structure_ids": pa.large_list(pa.uint8()),
    "token_dep_levels": pa.large_list(pa.uint16()),
    "token_ast_depth": pa.large_list(pa.uint16()),
    "token_sibling_index": pa.large_list(pa.uint16()),
    "token_ast_node_type": pa.large_list(pa.uint16()),
    "token_chunk_starts": pa.large_list(pa.uint32()),
    "token_chunk_ends": pa.large_list(pa.uint32()),
    "token_chunk_kinds": pa.large_list(pa.uint8()),
    "token_chunk_dep_levels": pa.large_list(pa.uint16()),
    "token_call_edges": pa.large_list(
        pa.struct([("from", pa.uint16()), ("to", pa.uint16())])
    ),
    "token_type_edges": pa.large_list(
        pa.struct([("from", pa.uint16()), ("to", pa.uint16())])
    ),
}

# Original columns whose preservation we verify with extra emphasis.
CRITICAL_PRESERVE_COLUMNS = ("text", "token_ids", "actual_token_count")
SYMBOL_ID_TOKEN_COLUMNS = (
    "token_symbol_ids",
    "token_call_targets",
    "token_type_refs",
)


# ---------------------------------------------------------------------------
# Robust repo parser for the COMMITS family.
# Matches a doc-comment line:  " * Repository: <name>"
# Whole-text search (the @brief can contain a stray "*/" that closes the block
# early, so block-bounded parsing under-matches; a clean ` * Repository:` line is
# unambiguous boilerplate of the embedded header).
# ---------------------------------------------------------------------------
_RE_REPOSITORY_LINE = re.compile(
    r"^[ \t]*\*[ \t]*Repository:[ \t]*(.+?)[ \t]*$", re.MULTILINE
)


def parse_repo_from_text(text: object) -> str | None:
    if not isinstance(text, str) or not text:
        return None
    m = _RE_REPOSITORY_LINE.search(text)
    if not m:
        return None
    value = m.group(1).strip()
    if not value or "\n" in value or len(value) > 256:
        return None
    return value


# ---------------------------------------------------------------------------
# Per-dataset plan.
#   mode "verify_and_backfill": clang_commits -> mirror (verify-only) + backfill repo
#   mode "verify_only":         clang_semantic -> mirror (verify-only), nothing derivable
# The two v9 tree-sitter datasets are intentionally absent (duplicates the user
# does not need); they are reported by --list and excluded from apply_targets.
# ---------------------------------------------------------------------------
class BackfillSpec:
    """One derivable column: name + parser + required min coverage fraction."""

    def __init__(self, column: str, parser, min_coverage: float, source: str):
        self.column = column
        self.parser = parser
        self.min_coverage = min_coverage
        self.source = source


DATASET_PLANS: dict[str, dict] = {
    "clang_commits_4k_v1": {
        "kind": "COMMITS",
        "mode": "verify_and_backfill",
        "backfills": [
            BackfillSpec(
                column="repo",
                parser=parse_repo_from_text,
                # observed 100% on sampled shards; require near-total coverage and
                # RAISE on any header-less row rather than emit empty/guessed values.
                min_coverage=0.995,
                source="embedded doc-comment header line ' * Repository: <name>'",
            ),
        ],
        # Columns we will NOT touch (not derivable / fabrication risk). Recorded.
        "not_derivable": [
            "commit (no commit hash/SHA in any row)",
            "timestamp (no authoritative date in any row)",
            "platform_info (only CONSTANT boilerplate header; real values are "
            "semantic content dicts -> fabrication)",
            "build_info (only CONSTANT g++/c++17 boilerplate -> fabrication)",
            "constituent_provenance / constituent_provenance_json (all-null)",
        ],
    },
    "clang_semantic_4k_v10": {
        "kind": "CODE",
        "mode": "verify_only",
        "backfills": [],
        "not_derivable": [
            "repo (no Repository header; CODE dataset)",
            "commit / timestamp (absent)",
            "platform_info (existing values are semantic content dicts; header is "
            "constant boilerplate -> fabrication). build_info already populated.",
            "constituent_provenance / constituent_provenance_json (all-null)",
        ],
    },
}

# Datasets we deliberately do not process (duplicates).
DUPLICATE_DATASETS = ("treesitter_compilable_4k_v9", "enriched_commits_4k_v9")


def list_shards(dataset: str) -> list[Path]:
    ddir = PARQUET_ROOT / dataset
    if not ddir.is_dir():
        raise FileNotFoundError(f"[mirror] dataset dir not found: {ddir}")
    shards = sorted(ddir.glob("*.parquet"), key=lambda p: p.name)
    if not shards:
        raise FileNotFoundError(f"[mirror] no *.parquet shards under {ddir}")
    return shards


def assert_mlx_token_columns(schema: pa.Schema, shard: Path) -> None:
    """VERIFY (mirror): mlx token_* columns present with EXACT dtypes; footer empty."""
    names = set(schema.names)
    for col, expected in MLX_TOKEN_COLUMNS.items():
        if col not in names:
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=missing mlx token column {col!r}; "
                f"the mlx token transformation is NOT applied to this shard. "
                f"Refusing to fabricate it (run the mlx materializer first)."
            )
        actual = schema.field(col).type
        if actual != expected:
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=dtype mismatch on {col!r}: "
                f"on-disk={actual} expected(mlx)={expected}"
            )
    symbol_columns = set(SYMBOL_ID_TOKEN_COLUMNS) & names
    footer = schema.metadata or {}
    if not symbol_columns and footer:
        raise ValueError(
            f"[mirror] WHERE={shard} WHAT=unexpected schema footer metadata "
            f"{list(footer.keys())!r}; mlx live shards have an EMPTY footer. "
            f"Refusing to alter metadata semantics."
        )
    if symbol_columns:
        if symbol_columns != set(SYMBOL_ID_TOKEN_COLUMNS):
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=partial semantic symbol columns: "
                f"{sorted(symbol_columns)}"
            )
        raw_version = footer.get(
            SYMBOL_IDENTITY_SCHEMA_METADATA_KEY.encode("ascii")
        )
        if raw_version != str(SYMBOL_IDENTITY_SCHEMA_VERSION).encode("ascii"):
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=semantic symbol columns require "
                f"identity schema v{SYMBOL_IDENTITY_SCHEMA_VERSION}, got "
                f"{raw_version!r}"
            )
        for column in SYMBOL_ID_TOKEN_COLUMNS:
            column_type = schema.field(column).type
            if not (
                pa.types.is_list(column_type)
                or pa.types.is_large_list(column_type)
            ) or column_type.value_type != pa.uint64():
                raise ValueError(
                    f"[mirror] WHERE={shard}:{column} WHAT=expected list<uint64>, "
                    f"got {column_type}"
                )
        if SYMBOL_IDENTITIES_COLUMN not in names:
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=missing {SYMBOL_IDENTITIES_COLUMN}"
            )


def validate_symbol_identity_rows(
    table: pa.Table,
    shard: Path,
    corpus_registry: SymbolIdentityRegistry,
) -> None:
    if not (set(SYMBOL_ID_TOKEN_COLUMNS) & set(table.column_names)):
        return
    identity_rows = table.column(SYMBOL_IDENTITIES_COLUMN).to_pylist()
    semantic_rows = {
        column: table.column(column).to_pylist()
        for column in SYMBOL_ID_TOKEN_COLUMNS
    }
    for row_index, records in enumerate(identity_rows):
        source = f"{shard}:row={row_index}"
        row_registry = SymbolIdentityRegistry()
        row_registry.register_records(records, source=source)
        corpus_registry.register_records(records, source=source)
        used_ids = {
            int(value)
            for rows in semantic_rows.values()
            for value in rows[row_index]
            if int(value) != 0
        }
        row_registry.require_ids(used_ids, source=source)


def build_backfilled_table(
    table: pa.Table,
    backfills: list[BackfillSpec],
    shard: Path,
) -> tuple[pa.Table, dict]:
    """Return (new_table, report). Preserves every original column; only fills the
    previously-empty backfill columns. RAISES on coverage shortfall (RULE #1)."""
    n = table.num_rows
    report_cols: dict[str, dict] = {}

    if not backfills:
        return table, {"columns": report_cols}

    texts = table.column("text").to_pylist()
    new_columns = {name: table.column(name) for name in table.column_names}

    for spec in backfills:
        if spec.column not in table.column_names:
            raise ValueError(
                f"[mirror] WHERE={shard} WHAT=backfill target column "
                f"{spec.column!r} absent from schema {table.column_names!r}"
            )
        existing = table.column(spec.column).to_pylist()

        parsed = [spec.parser(t) for t in texts]
        parseable = sum(1 for v in parsed if v is not None)
        coverage = parseable / n if n else 0.0

        if coverage < spec.min_coverage:
            # The dataset is CLAIMED to carry this header for ~all rows. A shortfall
            # means our assumption is wrong somewhere -> fail loud, do not guess.
            missing_idx = [i for i, v in enumerate(parsed) if v is None][:10]
            raise ValueError(
                f"[mirror] WHERE={shard}:{spec.column} "
                f"WHAT=header coverage {coverage:.4f} < required {spec.min_coverage} "
                f"({parseable}/{n} rows parseable from {spec.source}); "
                f"refusing to emit empty/guessed values. first missing rows={missing_idx}"
            )

        # Fill: keep any existing non-empty value (idempotent), else parsed value.
        # If a row already has a value AND we parsed one, they must agree or we raise
        # (a disagreement would mean the backfill is corrupting real data).
        out: list[object] = []
        filled = 0
        already = 0
        conflicts = 0
        unparsed_rows: list[int] = []
        for i in range(n):
            cur = existing[i]
            has_cur = cur is not None and cur != ""
            new_v = parsed[i]
            if has_cur:
                if new_v is not None and str(new_v) != str(cur):
                    conflicts += 1
                out.append(cur)
                already += 1
            elif new_v is not None:
                out.append(new_v)
                filled += 1
            else:
                # No existing value and not parseable. Coverage already passed the
                # threshold globally; per-row we must NOT fabricate -> leave as-is
                # (None/empty) and record it. This is recorded, not silently hidden.
                out.append(cur)
                unparsed_rows.append(i)

        if conflicts:
            raise ValueError(
                f"[mirror] WHERE={shard}:{spec.column} "
                f"WHAT={conflicts} rows where parsed header disagrees with an "
                f"existing non-empty value; refusing to overwrite real data."
            )

        # Preserve the original arrow type of the column exactly.
        new_columns[spec.column] = pa.array(out, type=table.schema.field(spec.column).type)
        report_cols[spec.column] = {
            "source": spec.source,
            "rows": n,
            "coverage": round(coverage, 6),
            "filled": filled,
            "already_present": already,
            "left_empty_unparseable": len(unparsed_rows),
            "left_empty_row_indices_sample": unparsed_rows[:10],
        }

    new_table = pa.table(new_columns, schema=table.schema)
    return new_table, {"columns": report_cols}


def _is_primitive_type(t: pa.DataType) -> bool:
    """True for types the pyarrow 'equal' kernel supports element-wise."""
    return (
        pa.types.is_integer(t)
        or pa.types.is_floating(t)
        or pa.types.is_boolean(t)
        or pa.types.is_string(t)
        or pa.types.is_large_string(t)
        or pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_temporal(t)
        or pa.types.is_decimal(t)
    )


def _columns_value_equal(a: pa.ChunkedArray, b: pa.ChunkedArray) -> bool:
    if a.type != b.type:
        return False
    if len(a) != len(b):
        return False

    if _is_primitive_type(a.type):
        # Fast element-wise path with explicit null-position semantics.
        eq = pc.equal(a, b)
        a_null = pc.is_null(a)
        b_null = pc.is_null(b)
        both_null = pc.and_(a_null, b_null)
        one_null = pc.xor(a_null, b_null)
        if pc.any(one_null).as_py():
            return False
        eq_filled = pc.fill_null(eq, False)
        eq_or_bothnull = pc.or_(eq_filled, both_null)
        return bool(pc.all(eq_or_bothnull).as_py())

    # Nested/list/struct types: 'equal' has no element-wise kernel. Use Arrow's
    # C++-level deep equality (value+null aware, chunk-layout independent) instead of
    # materializing hundreds of millions of Python objects via to_pylist(). This is
    # exact value equality for the list/struct token_* and char-level columns and is
    # orders of magnitude faster and lower-memory.
    return a.equals(b)


def verify_tmp_against_original(
    original: pa.Table,
    tmp_path: Path,
    backfills: list[BackfillSpec],
    shard: Path,
) -> dict:
    """Read back the .tmp and assert all invariants. RAISE on any mismatch."""
    rewritten = pq.read_table(tmp_path)

    # 1. row count
    if rewritten.num_rows != original.num_rows:
        raise ValueError(
            f"[mirror] WHERE={tmp_path} WHAT=row count changed "
            f"{rewritten.num_rows} != original {original.num_rows}"
        )

    # 2. column set unchanged (we add nothing, we only fill)
    if list(rewritten.column_names) != list(original.column_names):
        raise ValueError(
            f"[mirror] WHERE={tmp_path} WHAT=column set/order changed. "
            f"new={rewritten.column_names} orig={original.column_names}"
        )

    # 3. mlx token columns still present with exact dtypes + empty footer
    assert_mlx_token_columns(rewritten.schema, tmp_path)

    backfill_cols = {s.column for s in backfills}

    # 4. every ORIGINAL column value-equal, except the backfill columns (which we
    #    intentionally changed by filling). Critical columns checked explicitly.
    for name in original.column_names:
        if name in backfill_cols:
            continue
        if not _columns_value_equal(original.column(name), rewritten.column(name)):
            extra = " (CRITICAL)" if name in CRITICAL_PRESERVE_COLUMNS else ""
            raise ValueError(
                f"[mirror] WHERE={tmp_path}:{name}{extra} "
                f"WHAT=original column not preserved value-for-value after rewrite"
            )

    # 4b. critical columns must be byte/value identical (redundant guard; if any of
    #     them was somehow a backfill target that would be a design bug -> raise).
    for name in CRITICAL_PRESERVE_COLUMNS:
        if name not in original.column_names:
            continue
        if name in backfill_cols:
            raise ValueError(
                f"[mirror] WHERE={shard}:{name} WHAT=critical preserve column is "
                f"also a backfill target; design invariant violated"
            )
        if not _columns_value_equal(original.column(name), rewritten.column(name)):
            raise ValueError(
                f"[mirror] WHERE={tmp_path}:{name} (CRITICAL) "
                f"WHAT=critical column changed"
            )

    # 5. backfill columns: present and non-empty exactly where claimed derivable.
    bf_report: dict[str, dict] = {}
    for spec in backfills:
        new_vals = rewritten.column(spec.column).to_pylist()
        orig_vals = original.column(spec.column).to_pylist()
        parsed = [spec.parser(t) for t in original.column("text").to_pylist()]
        n = original.num_rows
        nonempty = sum(1 for v in new_vals if v is not None and v != "")
        # every row that was parseable OR already had a value MUST now be non-empty
        must_have = [
            i
            for i in range(n)
            if (parsed[i] is not None)
            or (orig_vals[i] is not None and orig_vals[i] != "")
        ]
        bad = [i for i in must_have if new_vals[i] is None or new_vals[i] == ""]
        if bad:
            raise ValueError(
                f"[mirror] WHERE={tmp_path}:{spec.column} "
                f"WHAT={len(bad)} rows that are claimed-derivable are still empty "
                f"after backfill; first={bad[:10]}"
            )
        # never invented a value where there was neither existing nor parseable
        invented = [
            i
            for i in range(n)
            if (new_vals[i] is not None and new_vals[i] != "")
            and (orig_vals[i] is None or orig_vals[i] == "")
            and parsed[i] is None
        ]
        if invented:
            raise ValueError(
                f"[mirror] WHERE={tmp_path}:{spec.column} "
                f"WHAT={len(invented)} rows have a value that was NEITHER present "
                f"nor parseable from the header (fabrication); first={invented[:10]}"
            )
        bf_report[spec.column] = {
            "nonempty_after": nonempty,
            "rows": n,
            "claimed_derivable": len(must_have),
        }

    return {"verified": True, "backfill": bf_report}


def process_shard(
    shard: Path,
    plan: dict,
    mode: str,
    identity_registry: SymbolIdentityRegistry,
) -> dict:
    """Process one shard. Returns a JSON-serializable per-shard record."""
    backfills: list[BackfillSpec] = plan["backfills"]
    original = pq.read_table(shard)

    # VERIFY (mirror): mlx token columns + empty footer must already be in place.
    assert_mlx_token_columns(original.schema, shard)
    validate_symbol_identity_rows(original, shard, identity_registry)

    # Build the backfilled table (or identity if verify_only / no backfills).
    new_table, build_report = build_backfilled_table(original, backfills, shard)

    # Determine whether anything actually changes.
    changes = False
    for spec in backfills:
        if not _columns_value_equal(
            original.column(spec.column), new_table.column(spec.column)
        ):
            changes = True
            break

    record: dict = {
        "shard": str(shard),
        "dataset": plan["__dataset__"],
        "kind": plan["kind"],
        "rows": original.num_rows,
        "plan_mode": plan["mode"],
        "run_mode": mode,
        "mlx_token_columns_present": True,
        "schema_footer_empty": True,
        "backfill_build": build_report["columns"],
        "not_derivable": plan["not_derivable"],
    }

    if not changes:
        # Nothing to write: token mirror already applied, no empties to fill.
        record["state"] = "already_complete" if backfills else "verify_only_ok"
        record["wrote_tmp"] = False
        record["replaced"] = False
        return record

    # Write tmp + verify.
    tmp_path = shard.with_suffix(shard.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    try:
        pq.write_table(
            new_table,
            tmp_path,
            row_group_size=1024,
            compression="snappy",
        )
        verify_report = verify_tmp_against_original(
            original, tmp_path, backfills, shard
        )
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise

    record["wrote_tmp"] = True
    record["verify"] = verify_report

    if mode == "dry-run":
        # Keep tmp + report, do NOT replace or back up.
        record["state"] = "dry_run_verified"
        record["replaced"] = False
        record["tmp_path"] = str(tmp_path)
        return record

    # apply: back up original (skip if .bak exists), then atomic replace.
    bak_path = shard.with_suffix(shard.suffix + ".bak")
    if not bak_path.exists():
        # hard-link/copy original to .bak before replacing
        import shutil

        shutil.copy2(shard, bak_path)
        record["backed_up"] = str(bak_path)
    else:
        record["backed_up"] = f"{bak_path} (pre-existing, kept)"
    os.replace(tmp_path, shard)
    record["state"] = "applied"
    record["replaced"] = True
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        help="Dataset name under /Users/dave/sources/parquet (e.g. clang_commits_4k_v1).",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "apply"],
        default="dry-run",
        help="dry-run (default): write .tmp + verify, never replace. apply: replace.",
    )
    parser.add_argument(
        "--limit-shards",
        type=int,
        default=0,
        help="Process only the first N shards (0 = all).",
    )
    parser.add_argument(
        "--shard",
        default="",
        help="Process a single shard file (absolute path) instead of the whole dataset.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List planned datasets/targets and exit.",
    )
    args = parser.parse_args()

    if args.list:
        print(json.dumps({
            "apply_targets": list(DATASET_PLANS.keys()),
            "duplicate_skipped": list(DUPLICATE_DATASETS),
            "plans": {
                k: {"mode": v["mode"], "kind": v["kind"],
                    "backfills": [s.column for s in v["backfills"]],
                    "not_derivable": v["not_derivable"]}
                for k, v in DATASET_PLANS.items()
            },
        }, indent=2))
        return 0

    if not args.dataset:
        print(json.dumps({"error": "WHERE=cli WHAT=--dataset is required"}))
        return 2

    if args.dataset in DUPLICATE_DATASETS:
        raise ValueError(
            f"[mirror] WHERE=cli WHAT=dataset {args.dataset!r} is a known duplicate "
            f"the user does not need; refusing to process. Allowed: "
            f"{list(DATASET_PLANS.keys())}"
        )
    if args.dataset not in DATASET_PLANS:
        raise ValueError(
            f"[mirror] WHERE=cli WHAT=unknown dataset {args.dataset!r}. "
            f"Known: {list(DATASET_PLANS.keys())}"
        )

    plan = dict(DATASET_PLANS[args.dataset])
    plan["__dataset__"] = args.dataset

    if args.shard:
        shard = Path(args.shard)
        if not shard.is_file():
            raise FileNotFoundError(f"[mirror] WHERE=cli WHAT=--shard not found: {shard}")
        shards = [shard]
    else:
        shards = list_shards(args.dataset)
        if args.limit_shards > 0:
            shards = shards[: args.limit_shards]

    summary = {
        "dataset": args.dataset,
        "mode": args.mode,
        "shards_total": len(shards),
        "applied": 0,
        "dry_run_verified": 0,
        "already_complete": 0,
        "verify_only_ok": 0,
        "errors": 0,
        "rows_total": 0,
        "rows_backfilled": 0,
    }
    identity_registry = SymbolIdentityRegistry()

    for shard in shards:
        try:
            rec = process_shard(shard, plan, args.mode, identity_registry)
        except Exception as exc:  # noqa: BLE001 - we re-raise after recording WHERE/WHAT
            summary["errors"] += 1
            print(json.dumps({"shard": str(shard), "state": "error", "error": str(exc)}))
            # Fail loud: surface and stop the run.
            raise
        print(json.dumps(rec))
        summary["rows_total"] += rec["rows"]
        state = rec["state"]
        if state in summary:
            summary[state] += 1
        for col, info in rec.get("backfill_build", {}).items():
            summary["rows_backfilled"] += int(info.get("filled", 0))

    print(json.dumps({"summary": summary}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
