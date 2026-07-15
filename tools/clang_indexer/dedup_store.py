"""Global, cross-process, resumable exact + near-duplicate store (SQLite-backed).

This module implements ONE shared dedup store used by BOTH the code indexer
(``index_project.py``) and the commit indexer (``process_commits.py``), so that
deduplication is global across every repo and across both the code and commit
streams. The store lives in a single SQLite database file (the ``--dedup-db``
path passed by the drivers).

Design (RULE #1 — fail loud, never silent):
  * SQLite is opened in WAL mode with ``synchronous=NORMAL`` so multiple
    processes can read/write the same file concurrently and safely.
  * Exact dedup: table ``exact(hash BLOB PRIMARY KEY)``. We ``INSERT OR IGNORE``
    the sha1 of the *normalized* body; ``cursor.rowcount == 0`` means the row
    already existed -> the document is an exact duplicate.
  * Near dedup: ``datasketch`` MinHashLSH (threshold=0.7, num_perm=256) over
    5-gram word shingles of the normalized body. The LSH bands are persisted in
    table ``lsh(band_id INT, band_hash BLOB, doc_id INT)`` indexed on
    ``(band_id, band_hash)`` so the index is global and resumable across runs
    and processes. We query that SQLite band index directly for candidates; we
    DO NOT preload the entire persisted LSH into every worker process. We also
    persist each accepted document's MinHash signature in
    ``minhash(doc_id INTEGER PRIMARY KEY, sig BLOB)`` so we can verify the TRUE
    Jaccard similarity (>= 0.7) of a candidate against its band neighbours
    BEFORE deciding it is a near-duplicate (LSH gives candidates only).
  * Staged mode: when ``stage_id`` is provided, exact/minhash/chunk claims are
    written to stage tables only. The parent pipeline promotes the stage after
    materialize/pack/append succeeds, or discards it on failure. Closing a
    staged store never promotes claims.

  If a ``--dedup-db`` path is given but SQLite cannot be opened, or ``datasketch``
  cannot be imported, we RAISE. There is no in-memory / degraded fallback when a
  db path is requested. The legacy in-RAM ``set()`` path lives in the callers and
  is used ONLY when no ``--dedup-db`` is given.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re
import sqlite3
import struct
import time
from typing import Iterable, Sequence

# --------------------------------------------------------------------------- #
# Normalization (NO alpha-rename): strip comments, collapse whitespace, trim.
# --------------------------------------------------------------------------- #
_BLOCK_COMMENT = re.compile(r"/\*.*?\*/", re.DOTALL)
_LINE_COMMENT = re.compile(r"//[^\n]*")
_WS_RUN = re.compile(r"\s+")


def normalize_body(text: str) -> str:
    """Normalize a document body for dedup hashing/shingling.

    Strips ``//`` line comments and ``/* ... */`` block comments, collapses every
    run of whitespace to a single space, and trims. Does NOT rename identifiers
    (no alpha-rename) so semantically-distinct code is not collapsed.
    """
    if not isinstance(text, str):
        raise TypeError(f"normalize_body expected str, got {type(text)!r}")
    # Order matters: remove block comments first (may span lines), then line
    # comments, then collapse whitespace.
    s = _BLOCK_COMMENT.sub(" ", text)
    s = _LINE_COMMENT.sub(" ", s)
    s = _WS_RUN.sub(" ", s)
    return s.strip()


def _sha1(text: str) -> bytes:
    return hashlib.sha1(text.encode("utf-8", errors="replace")).digest()


def _shingles(normalized: str, k: int = 5) -> list[str]:
    """5-gram word shingles over the normalized body."""
    words = normalized.split(" ")
    if len(words) < k:
        # Whole body is a single shingle when shorter than k words.
        return [" ".join(words)] if words and words[0] else []
    return [" ".join(words[i:i + k]) for i in range(len(words) - k + 1)]


def _sha1_tokens(token_ids: Sequence[int]) -> bytes:
    """Stable long hash of a token-id sequence.

    This is the CANONICAL exact-dedup key per the corrected design: the function
    (or commit doc) is hashed AFTER OUR tokenizer, so the whitespace sentinels
    (<SPACE>/<NL>) have already canonicalized formatting. Two functions that
    differ only in formatting tokenize to the SAME id sequence and collapse here.
    """
    if not isinstance(token_ids, (list, tuple)):
        raise TypeError(
            f"_sha1_tokens expected list/tuple of ints, got {type(token_ids)!r}"
        )
    # Pack as little-endian uint32; vocab is 65536 so 4 bytes/id is exact.
    buf = struct.pack(f"<{len(token_ids)}I", *(int(t) & 0xFFFFFFFF for t in token_ids))
    return hashlib.sha1(buf).digest()


def _token_shingles(token_ids: Sequence[int], k: int = 5) -> list[bytes]:
    """k-gram shingles over the token-id sequence (for MinHash near-dup).

    Each shingle is the packed bytes of ``k`` consecutive token ids. Operating on
    token ids (not text words) keeps the near-dup signal aligned with the same
    canonical, format-collapsed representation used by the exact hash.
    """
    ids = [int(t) & 0xFFFFFFFF for t in token_ids]
    if len(ids) < k:
        if not ids:
            return []
        return [struct.pack(f"<{len(ids)}I", *ids)]
    return [
        struct.pack(f"<{k}I", *ids[i:i + k])
        for i in range(len(ids) - k + 1)
    ]


# --------------------------------------------------------------------------- #
# Store.
# --------------------------------------------------------------------------- #
class DedupStore:
    """Cross-process SQLite-backed exact + near-duplicate store.

    Usage (CANONICAL token-id path — hash AFTER OUR tokenizer):
        store = DedupStore(db_path, near=True)
        if store.seen_exact_tokens(token_ids):   # exact dup -> drop
            ...
        elif store.seen_near_tokens(token_ids):  # near dup (Jaccard>=0.7) -> drop
            ...
        store.commit()               # periodically (e.g. every 1000 docs)

    The legacy text path (seen_exact/seen_near over normalize_body) is retained
    for callers that have not yet tokenized; the corrected dedup design uses the
    *_tokens methods so the dedup key is exactly sha1(token_ids).
    """

    NUM_PERM = 256
    THRESHOLD = 0.7
    SHINGLE_K = 5
    SQLITE_TIMEOUT_SECONDS = 300.0
    SQLITE_BUSY_TIMEOUT_MS = int(SQLITE_TIMEOUT_SECONDS * 1000)
    WRITE_RETRY_SLEEP_SECONDS = 0.05
    WRITE_RETRY_MAX_SLEEP_SECONDS = 2.0
    # Bound cross-process writer transactions. ONE dedup DB is shared by several
    # concurrent producers (the code + commit stages, and --repo-workers>1). Under
    # SQLite WAL each connection's UNCOMMITTED writes are invisible to every other
    # connection, so each near-dup reference doc (its minhash signature + lsh band
    # rows) that a writer buffers before committing is a window in which a
    # concurrent writer cannot see it and may accept the SAME near-duplicate. Exact
    # dups are still backstopped (the chunk_claims ledger commits immediately and
    # the exact table is INSERT-OR-IGNORE), but NEAR dups have no such backstop, so
    # this buffer size IS the per-writer cross-process near-dedup leak window. Keep
    # the default modest (128) so that window stays small; only raise
    # CPPMEGA_DEDUP_MAX_PENDING_BEFORE_COMMIT for single-writer runs, where the near
    # index is fully self-visible and larger transactions merely cut commit count.
    MAX_PENDING_BEFORE_COMMIT = int(
        os.environ.get("CPPMEGA_DEDUP_MAX_PENDING_BEFORE_COMMIT", "128")
    )
    WAL_AUTOCHECKPOINT_PAGES = int(
        os.environ.get("CPPMEGA_DEDUP_WAL_AUTOCHECKPOINT_PAGES", "10000")
    )
    JOURNAL_SIZE_LIMIT_BYTES = int(
        os.environ.get("CPPMEGA_DEDUP_JOURNAL_SIZE_LIMIT_BYTES", str(1024**3))
    )

    def __init__(
        self,
        db_path: str,
        *,
        near: bool = True,
        commit_every: int = 1000,
        stage_id: str | None = None,
        stage_db_path: str | None = None,
    ):
        if not db_path:
            raise ValueError("DedupStore requires a non-empty db_path")
        if stage_id is not None and not str(stage_id).strip():
            raise ValueError("DedupStore stage_id must be non-empty when provided")
        if stage_db_path is not None and stage_id is None:
            raise ValueError("DedupStore stage_db_path requires a stage_id")
        self.db_path = str(db_path)
        self.stage_db_path = str(stage_db_path) if stage_db_path is not None else None
        self.near_enabled = bool(near)
        self.commit_every = int(commit_every)
        self.stage_id = str(stage_id) if stage_id is not None else None
        self._pending = 0
        self._local_stage = self.stage_db_path is not None

        # FAIL LOUD: sqlite open failure raises (no in-memory fallback). In
        # local-stage mode the child process must not acquire a global writer
        # lock; it reads committed claims from the global DB and writes claims to
        # its private rwork stage DB. The parent promotes that DB only after the
        # parquet append/recompress has succeeded.
        self.conn = self._open_global_connection(read_only=self._local_stage)
        self.stage_conn = self.conn
        if self._local_stage:
            assert self.stage_db_path is not None
            Path(self.stage_db_path).parent.mkdir(parents=True, exist_ok=True)
            try:
                self.stage_conn = sqlite3.connect(
                    self.stage_db_path,
                    timeout=self.SQLITE_TIMEOUT_SECONDS,
                )
            except sqlite3.Error as exc:
                raise RuntimeError(
                    "DedupStore: failed to open local stage sqlite db at "
                    f"{self.stage_db_path!r}: {exc}"
                ) from exc

        # Cross-process safe pragmas.
        self.conn.execute(f"PRAGMA busy_timeout={self.SQLITE_BUSY_TIMEOUT_MS}")
        if self._local_stage:
            self.conn.execute("PRAGMA query_only=ON")
            # Local stage DBs are discardable rwork ledgers. The parent promotes
            # them only after parquet append succeeds; on subprocess failure the
            # whole stage file is thrown away. Avoid fsync/WAL checkpoint work
            # here while keeping the global committed DB on the durable path below.
            self.stage_conn.execute("PRAGMA journal_mode=MEMORY")
            self.stage_conn.execute("PRAGMA synchronous=OFF")
            self.stage_conn.execute(
                f"PRAGMA busy_timeout={self.SQLITE_BUSY_TIMEOUT_MS}"
            )
            self.stage_conn.execute(
                f"PRAGMA journal_size_limit={self.JOURNAL_SIZE_LIMIT_BYTES}"
            )
            self._init_stage_schema()
        else:
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")
            self.conn.execute(
                f"PRAGMA wal_autocheckpoint={self.WAL_AUTOCHECKPOINT_PAGES}"
            )
            self.conn.execute(
                f"PRAGMA journal_size_limit={self.JOURNAL_SIZE_LIMIT_BYTES}"
            )
            self._init_schema()
        if self.stage_id is not None:
            self._init_stage(self.stage_id)

        # Lazy datasketch objects; built only when near dedup is enabled.
        self._lsh = None
        self._minhash_cls = None
        self._minhash_permutations = None
        self._loaded_count = 0
        if self.near_enabled:
            self._init_near()

    # ----------------------------------------------------------------- #
    def _open_global_connection(self, *, read_only: bool) -> sqlite3.Connection:
        try:
            if read_only:
                uri = Path(self.db_path).resolve().as_uri() + "?mode=ro"
                return sqlite3.connect(
                    uri,
                    timeout=self.SQLITE_TIMEOUT_SECONDS,
                    uri=True,
                )
            return sqlite3.connect(
                self.db_path,
                timeout=self.SQLITE_TIMEOUT_SECONDS,
            )
        except sqlite3.Error as exc:
            mode = "read-only " if read_only else ""
            raise RuntimeError(
                f"DedupStore: failed to open {mode}sqlite db at "
                f"{self.db_path!r}: {exc}"
            ) from exc

    def _init_schema(self) -> None:
        c = self.conn
        self._execute_write("CREATE TABLE IF NOT EXISTS exact (hash BLOB PRIMARY KEY)")
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS lsh ("
            "band_id INTEGER NOT NULL, band_hash BLOB NOT NULL, doc_id INTEGER NOT NULL)"
        )
        self._execute_write(
            "CREATE INDEX IF NOT EXISTS lsh_band ON lsh (band_id, band_hash)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS minhash ("
            "doc_id INTEGER PRIMARY KEY, sig BLOB NOT NULL)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS dedup_meta (key TEXT PRIMARY KEY, val INTEGER)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS chunk_claims ("
            "namespace TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "claim_count INTEGER NOT NULL, "
            "PRIMARY KEY(namespace, hash))"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS dedup_stages ("
            "stage_id TEXT PRIMARY KEY, "
            "created_at REAL NOT NULL, "
            "next_doc_id INTEGER NOT NULL DEFAULT 0)"
        )
        stage_cols = {
            str(row[1])
            for row in self.conn.execute("PRAGMA table_info(dedup_stages)")
        }
        if "next_doc_id" not in stage_cols:
            self._execute_write(
                "ALTER TABLE dedup_stages "
                "ADD COLUMN next_doc_id INTEGER NOT NULL DEFAULT 0"
            )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS exact_stage ("
            "stage_id TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "PRIMARY KEY(stage_id, hash))"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS minhash_stage ("
            "stage_id TEXT NOT NULL, "
            "stage_doc_id INTEGER NOT NULL, "
            "sig BLOB NOT NULL, "
            "PRIMARY KEY(stage_id, stage_doc_id))"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS lsh_stage ("
            "stage_id TEXT NOT NULL, "
            "band_id INTEGER NOT NULL, "
            "band_hash BLOB NOT NULL, "
            "stage_doc_id INTEGER NOT NULL)"
        )
        self._execute_write(
            "CREATE INDEX IF NOT EXISTS lsh_stage_band "
            "ON lsh_stage (stage_id, band_id, band_hash)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS chunk_claims_stage ("
            "stage_id TEXT NOT NULL, "
            "namespace TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "claim_count INTEGER NOT NULL, "
            "PRIMARY KEY(stage_id, namespace, hash))"
        )
        # next doc_id counter for near-dup docs.
        self._execute_write(
            "INSERT OR IGNORE INTO dedup_meta (key, val) VALUES ('next_doc_id', 0)"
        )
        c.commit()

    def _init_stage_schema(self) -> None:
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS dedup_stages ("
            "stage_id TEXT PRIMARY KEY, "
            "created_at REAL NOT NULL, "
            "next_doc_id INTEGER NOT NULL DEFAULT 0)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS exact_stage ("
            "stage_id TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "PRIMARY KEY(stage_id, hash))"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS minhash_stage ("
            "stage_id TEXT NOT NULL, "
            "stage_doc_id INTEGER NOT NULL, "
            "sig BLOB NOT NULL, "
            "PRIMARY KEY(stage_id, stage_doc_id))"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS lsh_stage ("
            "stage_id TEXT NOT NULL, "
            "band_id INTEGER NOT NULL, "
            "band_hash BLOB NOT NULL, "
            "stage_doc_id INTEGER NOT NULL)"
        )
        self._execute_write(
            "CREATE INDEX IF NOT EXISTS lsh_stage_band "
            "ON lsh_stage (stage_id, band_id, band_hash)"
        )
        self._execute_write(
            "CREATE TABLE IF NOT EXISTS chunk_claims_stage ("
            "stage_id TEXT NOT NULL, "
            "namespace TEXT NOT NULL, "
            "hash BLOB NOT NULL, "
            "claim_count INTEGER NOT NULL, "
            "PRIMARY KEY(stage_id, namespace, hash))"
        )
        self.stage_conn.commit()

    def _execute_write(
        self,
        sql: str,
        params: tuple = (),
        *,
        conn: sqlite3.Connection | None = None,
        db_path: str | None = None,
    ):
        """Execute a SQLite write with bounded retry on cross-process writer locks.

        The corpus conveyor runs code and commit stages concurrently against one
        global dedup DB. SQLite WAL allows this, but only one writer can commit at
        a time. We keep fail-loud semantics: lock contention is retried for a
        bounded window, then raised with WHERE context instead of falling back.
        """
        target = conn or (self.stage_conn if self._local_stage else self.conn)
        target_path = db_path or (
            self.stage_db_path if self._local_stage else self.db_path
        )
        deadline = time.monotonic() + self.SQLITE_TIMEOUT_SECONDS
        sleep_s = self.WRITE_RETRY_SLEEP_SECONDS
        while True:
            try:
                return target.execute(sql, params)
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "DedupStore: sqlite write remained locked after "
                        f"{self.SQLITE_TIMEOUT_SECONDS:.0f}s at {target_path!r}"
                    ) from exc
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, self.WRITE_RETRY_MAX_SLEEP_SECONDS)

    def _begin_immediate(
        self,
        *,
        conn: sqlite3.Connection | None = None,
        db_path: str | None = None,
    ) -> None:
        target = conn or (self.stage_conn if self._local_stage else self.conn)
        target_path = db_path or (
            self.stage_db_path if self._local_stage else self.db_path
        )
        deadline = time.monotonic() + self.SQLITE_TIMEOUT_SECONDS
        sleep_s = self.WRITE_RETRY_SLEEP_SECONDS
        while True:
            try:
                target.execute("BEGIN IMMEDIATE")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc).lower():
                    raise
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "DedupStore: sqlite transaction remained locked after "
                        f"{self.SQLITE_TIMEOUT_SECONDS:.0f}s at {target_path!r}"
                    ) from exc
                time.sleep(sleep_s)
                sleep_s = min(sleep_s * 1.5, self.WRITE_RETRY_MAX_SLEEP_SECONDS)

    def _init_stage(self, stage_id: str) -> None:
        # A stage id is the pipeline unit identity (code repo or commit range).
        # Stale cleanup is explicit (discard_stage) at subprocess entry. A single
        # indexer process may open several DedupStore handles for root/build/chunk
        # claims, so constructor-time cleanup would erase earlier staged claims.
        self._execute_write(
            "INSERT OR IGNORE INTO dedup_stages(stage_id, created_at, next_doc_id) "
            "VALUES (?, ?, 0)",
            (stage_id, time.time()),
        )
        self.stage_conn.commit()

    def _discard_stage_rows(self, stage_id: str) -> None:
        for table in (
            "lsh_stage",
            "minhash_stage",
            "exact_stage",
            "chunk_claims_stage",
            "dedup_stages",
        ):
            self._execute_write(f"DELETE FROM {table} WHERE stage_id=?", (stage_id,))

    def _init_near(self) -> None:
        # FAIL LOUD: datasketch import failure raises (no fallback).
        try:
            from datasketch import MinHash, MinHashLSH
        except Exception as exc:  # noqa: BLE001 - re-raise loudly
            raise RuntimeError(
                "DedupStore: datasketch import failed; near-dup requires datasketch "
                f"(pip install datasketch). Underlying error: {exc}"
            ) from exc

        self._minhash_cls = MinHash
        # Keep a tiny MinHashLSH instance only for its deterministic banding
        # parameters (b, r). Persisted LSH candidates are queried from SQLite by
        # band hash; preloading a multi-GB minhash table into every worker was
        # the conveyor memory/I/O bottleneck.
        self._lsh = MinHashLSH(threshold=self.THRESHOLD, num_perm=self.NUM_PERM)
        # datasketch.MinHash otherwise regenerates random permutation arrays for
        # every document. Reuse the exact default-seed permutations once.
        self._minhash_permutations = MinHash(num_perm=self.NUM_PERM).permutations

    # ----------------------------------------------------------------- #
    # MinHash (de)serialization.
    # ----------------------------------------------------------------- #
    def _new_minhash(self):
        assert self._minhash_cls is not None
        return self._minhash_cls(
            num_perm=self.NUM_PERM,
            permutations=self._minhash_permutations,
        )

    def _build_minhash(self, normalized: str):
        mh = self._new_minhash()
        for sh in _shingles(normalized, self.SHINGLE_K):
            mh.update(sh.encode("utf-8", errors="replace"))
        return mh

    def _build_minhash_tokens(self, token_ids: Sequence[int]):
        mh = self._new_minhash()
        for sh in _token_shingles(token_ids, self.SHINGLE_K):
            mh.update(sh)
        return mh

    @staticmethod
    def _minhash_to_blob(mh) -> bytes:
        arr = mh.hashvalues  # numpy uint64 array, length num_perm
        return struct.pack(f"<{len(arr)}Q", *(int(x) for x in arr))

    def _minhash_from_blob(self, blob: bytes):
        n = len(blob) // 8
        vals = struct.unpack(f"<{n}Q", blob)
        import numpy as np
        mh = self._minhash_cls(
            num_perm=self.NUM_PERM,
            hashvalues=np.array(vals, dtype=np.uint64),
            permutations=self._minhash_permutations,
        )
        return mh

    # ----------------------------------------------------------------- #
    # Public API.
    # ----------------------------------------------------------------- #
    def seen_exact(self, text: str) -> bool:
        """Return True if this body's normalized sha1 was already seen.

        Inserts the hash atomically (INSERT OR IGNORE); a 0 rowcount means the
        row already existed -> exact duplicate.
        """
        h = _sha1(normalize_body(text))
        if self.stage_id is not None:
            return self._seen_exact_staged(h)
        cur = self._execute_write(
            "INSERT OR IGNORE INTO exact (hash) VALUES (?)", (h,)
        )
        self._mark_pending()
        return cur.rowcount == 0

    def seen_near(self, text: str) -> bool:
        """Return True if a near-duplicate (true Jaccard >= 0.7) already exists.

        On a non-duplicate, the document's MinHash signature and LSH bands are
        persisted (global + resumable) so future documents dedup against it.
        """
        if not self.near_enabled:
            raise RuntimeError("seen_near called but near dedup is disabled")
        normalized = normalize_body(text)
        mh = self._build_minhash(normalized)

        if self._has_near_duplicate(mh):
            return True

        # Not a near-dup: persist signature + bands so it becomes a reference.
        if self.stage_id is not None:
            self._persist_staged_minhash(mh)
            self._mark_pending()
            return False
        doc_id = self._next_doc_id()
        sig = self._minhash_to_blob(mh)
        self._execute_write(
            "INSERT INTO minhash (doc_id, sig) VALUES (?, ?)", (doc_id, sig)
        )
        self._persist_bands(doc_id, mh)
        self._mark_pending()
        return False

    # ----------------------------------------------------------------- #
    # Token-id-keyed API (CANONICAL per corrected dedup design).
    #
    # These hash the function/commit-doc AFTER OUR TOKENIZER: the exact key is
    # sha1(token_ids) and the near MinHash shingles are over token-id n-grams.
    # The whitespace sentinels (<SPACE>/<NL>) have already canonicalized format
    # in the id sequence, so format-only differences collapse to one row. Both
    # the exact set and the near LSH/minhash tables are SHARED with the text API
    # (same tables), so the store is global across code (token-id path) and
    # commits (token-id path) within one db file.
    # ----------------------------------------------------------------- #
    def seen_exact_tokens(self, token_ids: Sequence[int]) -> bool:
        """Return True if sha1(token_ids) was already seen (exact duplicate)."""
        h = _sha1_tokens(token_ids)
        if self.stage_id is not None:
            return self._seen_exact_staged(h)
        cur = self._execute_write(
            "INSERT OR IGNORE INTO exact (hash) VALUES (?)", (h,)
        )
        self._mark_pending()
        return cur.rowcount == 0

    def seen_near_tokens(self, token_ids: Sequence[int]) -> bool:
        """Return True if a near-dup (true Jaccard >= 0.7) over token-id shingles
        already exists. Persists this doc's signature + bands otherwise."""
        if not self.near_enabled:
            raise RuntimeError("seen_near_tokens called but near dedup is disabled")
        mh = self._build_minhash_tokens(token_ids)

        if self._has_near_duplicate(mh):
            return True

        if self.stage_id is not None:
            self._persist_staged_minhash(mh)
            self._mark_pending()
            return False
        doc_id = self._next_doc_id()
        sig = self._minhash_to_blob(mh)
        self._execute_write(
            "INSERT INTO minhash (doc_id, sig) VALUES (?, ?)", (doc_id, sig)
        )
        self._persist_bands(doc_id, mh)
        self._mark_pending()
        return False

    def claim_chunk_tokens(
        self,
        token_ids: Sequence[int],
        *,
        namespace: str = "train_chunk",
        max_count: int = 1,
    ) -> bool:
        """Claim one semantic training chunk by its tokenized body.

        This is separate from the ``exact``/``minhash`` dedup tables. Exact/near
        dedup decides whether a function may be emitted as a root document;
        chunk claims decide whether that already-tokenized function/class/type
        body may appear anywhere in the training stream. The unit of ownership is
        still the caller-provided semantic chunk; the hash is only the SQLite key.

        Returns True when this call successfully claimed one slot. Returns False
        when the tokenized chunk has already reached ``max_count`` in ``namespace``.
        The write is committed immediately so concurrent conveyor processes see
        the claim before they assemble overlapping 1k/2k/4k/8k buckets.
        """
        if max_count < 1:
            raise ValueError("claim_chunk_tokens requires max_count >= 1")
        if not namespace:
            raise ValueError("claim_chunk_tokens requires a non-empty namespace")
        h = _sha1_tokens(token_ids)
        if self.stage_id is not None:
            committed = self.conn.execute(
                "SELECT claim_count FROM chunk_claims WHERE namespace=? AND hash=?",
                (namespace, h),
            ).fetchone()
            staged = self.stage_conn.execute(
                "SELECT claim_count FROM chunk_claims_stage "
                "WHERE stage_id=? AND namespace=? AND hash=?",
                (self.stage_id, namespace, h),
            ).fetchone()
            count = (int(committed[0]) if committed else 0) + (
                int(staged[0]) if staged else 0
            )
            if count >= int(max_count):
                return False
            self._execute_write(
                "INSERT INTO chunk_claims_stage(stage_id, namespace, hash, claim_count) "
                "VALUES (?, ?, ?, 1) "
                "ON CONFLICT(stage_id, namespace, hash) DO UPDATE SET "
                "claim_count = claim_count + 1",
                (self.stage_id, namespace, h),
            )
            self._mark_pending()
            return True
        cur = self._execute_write(
            "INSERT INTO chunk_claims(namespace, hash, claim_count) "
            "VALUES (?, ?, 1) "
            "ON CONFLICT(namespace, hash) DO UPDATE SET "
            "claim_count = claim_count + 1 "
            "WHERE claim_count < ?",
            (namespace, h, int(max_count)),
        )
        # Claims are coordination records, not buffered dedup rows. Commit them
        # immediately so other WAL connections see the slot before emitting docs.
        self.commit()
        return cur.rowcount == 1

    # ----------------------------------------------------------------- #
    def _seen_exact_staged(self, h: bytes) -> bool:
        assert self.stage_id is not None
        if self.conn.execute("SELECT 1 FROM exact WHERE hash=?", (h,)).fetchone():
            return True
        if self.stage_conn.execute(
            "SELECT 1 FROM exact_stage WHERE stage_id=? AND hash=?",
            (self.stage_id, h),
        ).fetchone():
            return True
        self._execute_write(
            "INSERT OR IGNORE INTO exact_stage(stage_id, hash) VALUES (?, ?)",
            (self.stage_id, h),
        )
        self._mark_pending()
        return False

    def _has_near_duplicate(self, mh) -> bool:
        for cand_doc_id in self._query_lsh_candidates(mh):
            cand_mh = self._load_minhash(cand_doc_id)
            if cand_mh is None:
                continue
            if mh.jaccard(cand_mh) >= self.THRESHOLD:
                return True
        if self.stage_id is not None:
            for stage_doc_id in self._query_stage_lsh_candidates(mh):
                cand_mh = self._load_stage_minhash(stage_doc_id)
                if cand_mh is None:
                    continue
                if mh.jaccard(cand_mh) >= self.THRESHOLD:
                    return True
        return False

    def _load_minhash(self, doc_id: int):
        row = self.conn.execute(
            "SELECT sig FROM minhash WHERE doc_id=?", (doc_id,)
        ).fetchone()
        if row is None:
            return None
        return self._minhash_from_blob(row[0])

    def _band_hashes(self, mh) -> Iterable[tuple[int, bytes]]:
        """Yield SQLite-persisted LSH band keys for ``mh``."""
        b = self._lsh.b
        r = self._lsh.r
        hv = mh.hashvalues
        for band_id in range(b):
            start = band_id * r
            band_slice = hv[start:start + r]
            yield band_id, hashlib.sha1(
                struct.pack(f"<{len(band_slice)}Q", *(int(x) for x in band_slice))
            ).digest()

    def _query_lsh_candidates(self, mh) -> Iterable[int]:
        """Query persisted LSH bands without preloading the whole DB."""
        seen: set[int] = set()
        for band_id, band_hash in self._band_hashes(mh):
            cur = self.conn.execute(
                "SELECT doc_id FROM lsh WHERE band_id=? AND band_hash=?",
                (band_id, band_hash),
            )
            for (doc_id,) in cur:
                doc_id = int(doc_id)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield doc_id

    def _query_stage_lsh_candidates(self, mh) -> Iterable[int]:
        """Query this stage's unpromoted LSH bands."""
        assert self.stage_id is not None
        seen: set[int] = set()
        for band_id, band_hash in self._band_hashes(mh):
            cur = self.stage_conn.execute(
                "SELECT stage_doc_id FROM lsh_stage "
                "WHERE stage_id=? AND band_id=? AND band_hash=?",
                (self.stage_id, band_id, band_hash),
            )
            for (doc_id,) in cur:
                doc_id = int(doc_id)
                if doc_id in seen:
                    continue
                seen.add(doc_id)
                yield doc_id

    def _load_stage_minhash(self, stage_doc_id: int):
        assert self.stage_id is not None
        row = self.stage_conn.execute(
            "SELECT sig FROM minhash_stage WHERE stage_id=? AND stage_doc_id=?",
            (self.stage_id, int(stage_doc_id)),
        ).fetchone()
        if row is None:
            return None
        return self._minhash_from_blob(row[0])

    def _persist_bands(self, doc_id: int, mh) -> None:
        """Persist this doc's LSH band hashes into the lsh table."""
        for band_id, band_hash in self._band_hashes(mh):
            self._execute_write(
                "INSERT INTO lsh (band_id, band_hash, doc_id) VALUES (?, ?, ?)",
                (band_id, band_hash, doc_id),
            )

    def _persist_staged_minhash(self, mh) -> None:
        assert self.stage_id is not None
        stage_doc_id = self._next_stage_doc_id()
        self._execute_write(
            "INSERT INTO minhash_stage(stage_id, stage_doc_id, sig) VALUES (?, ?, ?)",
            (self.stage_id, stage_doc_id, self._minhash_to_blob(mh)),
        )
        for band_id, band_hash in self._band_hashes(mh):
            self._execute_write(
                "INSERT INTO lsh_stage(stage_id, band_id, band_hash, stage_doc_id) "
                "VALUES (?, ?, ?, ?)",
                (self.stage_id, band_id, band_hash, stage_doc_id),
            )

    def _next_stage_doc_id(self) -> int:
        assert self.stage_id is not None
        cur = self._execute_write(
            "UPDATE dedup_stages SET next_doc_id = next_doc_id + 1 "
            "WHERE stage_id=?",
            (self.stage_id,),
        )
        if cur.rowcount != 1:
            self._init_stage(self.stage_id)
            cur = self._execute_write(
                "UPDATE dedup_stages SET next_doc_id = next_doc_id + 1 "
                "WHERE stage_id=?",
                (self.stage_id,),
            )
        if cur.rowcount != 1:
            raise RuntimeError(
                f"DedupStore: failed to advance staged doc id for {self.stage_id!r}"
            )
        row = self.stage_conn.execute(
            "SELECT next_doc_id FROM dedup_stages WHERE stage_id=?",
            (self.stage_id,),
        ).fetchone()
        if row is None:
            raise RuntimeError(
                f"DedupStore: failed to read staged doc id for {self.stage_id!r}"
            )
        return int(row[0])

    def _next_doc_id(self) -> int:
        cur = self._execute_write(
            "UPDATE dedup_meta SET val = val + 1 WHERE key='next_doc_id'"
        )
        if cur.rowcount != 1:
            raise RuntimeError("DedupStore: failed to advance next_doc_id counter")
        row = self.conn.execute(
            "SELECT val FROM dedup_meta WHERE key='next_doc_id'"
        ).fetchone()
        return int(row[0])

    def _mark_pending(self) -> None:
        self._pending += 1
        threshold = min(self.commit_every, self.MAX_PENDING_BEFORE_COMMIT)
        if self._pending >= threshold:
            self.commit()

    def promote_stage(self, stage_id: str | None = None) -> None:
        """Atomically promote staged claims into the committed global ledger.

        Call this only after the pipeline unit's downstream materialize/pack/append
        has succeeded. Closing a staged writer intentionally does NOT promote:
        failed units leave only discardable staging rows.
        """
        sid = stage_id or self.stage_id
        if sid is None:
            raise ValueError("promote_stage requires a stage_id")
        if self._local_stage:
            assert self.stage_db_path is not None
            self.commit()
            self.promote_stage_from_db(self.db_path, self.stage_db_path, sid)
            return
        self.commit()
        self._begin_immediate()
        try:
            self.conn.execute(
                "INSERT OR IGNORE INTO exact(hash) "
                "SELECT hash FROM exact_stage WHERE stage_id=?",
                (sid,),
            )
            staged_minhashes = list(
                self.conn.execute(
                    "SELECT stage_doc_id, sig FROM minhash_stage "
                    "WHERE stage_id=? ORDER BY stage_doc_id",
                    (sid,),
                )
            )
            for stage_doc_id, sig in staged_minhashes:
                self.conn.execute(
                    "UPDATE dedup_meta SET val = val + 1 WHERE key='next_doc_id'"
                )
                row = self.conn.execute(
                    "SELECT val FROM dedup_meta WHERE key='next_doc_id'"
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "DedupStore: failed to read next_doc_id during stage promote"
                    )
                doc_id = int(row[0])
                self.conn.execute(
                    "INSERT INTO minhash(doc_id, sig) VALUES (?, ?)",
                    (doc_id, sig),
                )
                self.conn.execute(
                    "INSERT INTO lsh(band_id, band_hash, doc_id) "
                    "SELECT band_id, band_hash, ? FROM lsh_stage "
                    "WHERE stage_id=? AND stage_doc_id=?",
                    (doc_id, sid, int(stage_doc_id)),
                )
            self.conn.execute(
                "INSERT INTO chunk_claims(namespace, hash, claim_count) "
                "SELECT namespace, hash, claim_count FROM chunk_claims_stage "
                "WHERE stage_id=? "
                "ON CONFLICT(namespace, hash) DO UPDATE SET "
                "claim_count = chunk_claims.claim_count + excluded.claim_count",
                (sid,),
            )
            self._discard_stage_rows(sid)
            self.conn.commit()
            self._pending = 0
        except Exception:
            self.conn.rollback()
            raise

    @classmethod
    def promote_stage_in_db(cls, db_path: str, stage_id: str) -> None:
        store = cls(db_path, near=False)
        try:
            store.promote_stage(stage_id)
        finally:
            store.close()

    @classmethod
    def promote_stage_from_db(
        cls,
        db_path: str,
        stage_db_path: str,
        stage_id: str,
    ) -> None:
        """Promote one local rwork stage DB into the global committed ledger."""
        if not stage_id:
            raise ValueError("promote_stage_from_db requires a stage_id")
        if not os.path.exists(stage_db_path):
            raise FileNotFoundError(f"stage db missing: {stage_db_path}")
        store = cls(db_path, near=False)
        attached = False
        try:
            store.conn.execute("ATTACH DATABASE ? AS stage", (stage_db_path,))
            attached = True
            store._begin_immediate(conn=store.conn, db_path=db_path)
            try:
                store.conn.execute(
                    "INSERT OR IGNORE INTO exact(hash) "
                    "SELECT hash FROM stage.exact_stage WHERE stage_id=?",
                    (stage_id,),
                )
                row = store.conn.execute(
                    "SELECT val FROM dedup_meta WHERE key='next_doc_id'"
                ).fetchone()
                if row is None:
                    raise RuntimeError(
                        "DedupStore: failed to read next_doc_id during "
                        "local stage promote"
                    )
                base_doc_id = int(row[0])
                store.conn.execute(
                    "CREATE TEMP TABLE IF NOT EXISTS _dedup_stage_doc_map ("
                    "stage_doc_id INTEGER PRIMARY KEY, "
                    "doc_id INTEGER NOT NULL UNIQUE"
                    ")"
                )
                store.conn.execute("DELETE FROM _dedup_stage_doc_map")
                store.conn.execute(
                    "INSERT INTO _dedup_stage_doc_map(stage_doc_id, doc_id) "
                    "SELECT stage_doc_id, ? + ROW_NUMBER() OVER "
                    "(ORDER BY stage_doc_id) "
                    "FROM stage.minhash_stage WHERE stage_id=?",
                    (base_doc_id, stage_id),
                )
                store.conn.execute(
                    "UPDATE dedup_meta SET val = val + "
                    "(SELECT COUNT(*) FROM _dedup_stage_doc_map) "
                    "WHERE key='next_doc_id'"
                )
                store.conn.execute(
                    "INSERT INTO minhash(doc_id, sig) "
                    "SELECT m.doc_id, s.sig "
                    "FROM stage.minhash_stage AS s "
                    "JOIN _dedup_stage_doc_map AS m "
                    "ON m.stage_doc_id = s.stage_doc_id "
                    "WHERE s.stage_id=? "
                    "ORDER BY s.stage_doc_id",
                    (stage_id,),
                )
                store.conn.execute(
                    "INSERT INTO lsh(band_id, band_hash, doc_id) "
                    "SELECT l.band_id, l.band_hash, m.doc_id "
                    "FROM stage.lsh_stage AS l "
                    "JOIN _dedup_stage_doc_map AS m "
                    "ON m.stage_doc_id = l.stage_doc_id "
                    "WHERE l.stage_id=?",
                    (stage_id,),
                )
                store.conn.execute(
                    "INSERT INTO chunk_claims(namespace, hash, claim_count) "
                    "SELECT namespace, hash, claim_count "
                    "FROM stage.chunk_claims_stage WHERE stage_id=? "
                    "ON CONFLICT(namespace, hash) DO UPDATE SET "
                    "claim_count = chunk_claims.claim_count + excluded.claim_count",
                    (stage_id,),
                )
                for table in (
                    "lsh_stage",
                    "minhash_stage",
                    "exact_stage",
                    "chunk_claims_stage",
                    "dedup_stages",
                ):
                    store.conn.execute(
                        f"DELETE FROM stage.{table} WHERE stage_id=?",
                        (stage_id,),
                    )
                store.conn.commit()
            except Exception:
                store.conn.rollback()
                raise
        finally:
            if attached:
                store.conn.execute("DETACH DATABASE stage")
            store.close()

    def discard_current_stage(self) -> None:
        if self.stage_id is None:
            raise ValueError("discard_current_stage requires a staged DedupStore")
        self.discard_stage(
            self.db_path,
            self.stage_id,
            stage_db_path=self.stage_db_path,
        )

    @classmethod
    def discard_stage(
        cls,
        db_path: str,
        stage_id: str,
        *,
        stage_db_path: str | None = None,
    ) -> None:
        if stage_db_path is not None:
            if not os.path.exists(stage_db_path):
                return
            conn = sqlite3.connect(
                stage_db_path,
                timeout=cls.SQLITE_TIMEOUT_SECONDS,
            )
            try:
                conn.execute(f"PRAGMA busy_timeout={cls.SQLITE_BUSY_TIMEOUT_MS}")
                for table in (
                    "lsh_stage",
                    "minhash_stage",
                    "exact_stage",
                    "chunk_claims_stage",
                    "dedup_stages",
                ):
                    try:
                        conn.execute(
                            f"DELETE FROM {table} WHERE stage_id=?",
                            (stage_id,),
                        )
                    except sqlite3.OperationalError as exc:
                        if "no such table" in str(exc).lower():
                            continue
                        raise
                conn.commit()
            finally:
                conn.close()
            return
        store = cls(db_path, near=False)
        try:
            store._discard_stage_rows(stage_id)
            store.conn.commit()
        finally:
            store.close()

    def commit(self) -> None:
        self.stage_conn.commit()
        if self.stage_conn is not self.conn:
            self.conn.commit()
        self._pending = 0

    def close(self) -> None:
        try:
            self.commit()
        finally:
            if self.stage_conn is not self.conn:
                self.stage_conn.close()
            self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
