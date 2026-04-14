# megacpp data preparation pipeline

End-to-end recipe for producing the Megatron-format dataset that
`scripts/remote_smoke_h200_dsa_9_4_m.sh` (and every other NAM56R launcher)
reads via `--data-path`.

All stages live under `scripts/data/` and are orchestrated by
`scripts/data/prepare_data.sh`.

```
scripts/data/
├── prepare_data.sh                           # master dispatcher
├── prepare_download_megacpp.sh               # stage 1: clone C/C++ repos
├── prepare_tokenize_megacpp.py               # stage 2: clang index + BPE tokenize -> parquet
├── prepare_stream_jsonl_to_parquet_megacpp.py  #   helper (self-contained)
├── prepare_format_megacpp.py                 # stage 3: parquet -> Megatron .bin/.idx
├── prepare_cache_megacpp.py                  # stage 4: warm index cache
└── verify_dataset_megacpp.py                 # stage 5: sanity-check
```

---

## Source data

**Corpus:** eight public C/C++ repositories cloned shallow:

| repo | ref | purpose |
|---|---|---|
| `llvm/llvm-project` | `llvmorg-19.1.0` | modern C++, compilers |
| `boostorg/boost`   | `boost-1.86.0` (+submodules) | template-heavy C++ |
| `torvalds/linux`   | `v6.10` | C, systems |
| `fmtlib/fmt`       | `11.0.0` | small high-quality C++ |
| `google/googletest`| `v1.15.0` | test patterns |
| `abseil/abseil-cpp`| tip        | Google C++ commons |
| `facebook/folly`   | tip        | Facebook C++ |
| `grpc/grpc`        | `v1.67.0` | large C++ service framework |

**License + provenance:** all sources are public on GitHub under their
respective upstream licenses (Apache 2.0 for LLVM/abseil/grpc, BSL-1.0 for
Boost, GPL-2.0 for Linux kernel headers, MIT for fmt/googletest,
Apache 2.0 for folly). No credentials required to clone.

**Size:** shallow clones total ~15 GB on disk; after filtering to
C/C++/header files the usable corpus is several hundred thousand files.
Exact token count depends on tokenizer + filtering — `verify_dataset_megacpp.py`
prints it at the end.

---

## Tokenizer

**Vocab:** 131 072 tokens (aligned with `--vocab-size` in training launchers
and with the nanochat reference).

**Style:** hybrid fixed C++ vocabulary + learned BPE with BERT-style
whitespace handling. Implemented in nanochat's `nanochat/cpp_tokenizer.py`
(HuggingFace `tokenizers` backend).

**Artifact:** `tokenizer.json` (~2.2 MiB).
`prepare_tokenize_megacpp.py` copies it from
`${MEGACPP_NANOCHAT_ROOT}/tokenizer.json` into
`${MEGACPP_DATA_ROOT}/tokenizer/tokenizer.json` so the launcher's
`--tokenizer-model "${REMOTE_ROOT}/data/tokenizer"` can find it.

> **Note:** the tokenizer artifact is *not* vendored inside cppmega — it
> is a binary artifact owned by the nanochat repo. Set
> `MEGACPP_NANOCHAT_ROOT` to point at a local nanochat checkout.

---

## Pipeline

### Stage 1 — download
```bash
bash scripts/data/prepare_data.sh download
# or directly:
bash scripts/data/prepare_download_megacpp.sh
```
Shallow-clones the eight repos into `${MEGACPP_DATA_ROOT}/cpp_raw/`.
Idempotent — re-running skips existing directories.

### Stage 2 — tokenize (requires nanochat)
```bash
export MEGACPP_NANOCHAT_ROOT=/path/to/nanochat
python scripts/data/prepare_tokenize_megacpp.py
```
Dispatches to `nanochat/scripts/data/run_clang_pipeline.sh`, which:
1. Runs libclang-based semantic indexing over each project.
2. Emits enriched JSONL (one doc per semantic chunk, ≤4096 tokens).
3. Tokenizes with the hybrid BPE tokenizer.
4. Streams into parquet shards of 50k docs each, plus a `val_shard.parquet`.
5. Writes `_COMPLETE` sentinel.

Output: `${MEGACPP_DATA_ROOT}/parquet/clang_semantic_4k_v10/`.

**Why the nanochat dependency?** The clang indexer and tokenizer modules
(`nanochat/tools/clang_indexer/`, `nanochat/nanochat/cpp_tokenizer.py`,
`nanochat/nanochat/tokenized_enriched*`) are several thousand lines each
and pull in libclang bindings. Vendoring them into cppmega would duplicate
a maintained upstream. See `scripts/data/prepare_tokenize_megacpp.py`
header for the exact file list.

### Stage 3 — format (parquet → Megatron `.bin`/`.idx`)
```bash
python scripts/data/prepare_format_megacpp.py
```
Writes both splits:
- `${MEGACPP_DATA_ROOT}/megatron/clang_semantic_4k_v10_train.{bin,idx}`
- `${MEGACPP_DATA_ROOT}/megatron/clang_semantic_4k_v10_valid.{bin,idx}`

Uses `uint32` tokens (vocab=131072 > 65535 so `uint16` is invalid).
Implementation lives in `scripts/data_prep_parquet_to_megatron.py` which
prefers `megatron.core.datasets.indexed_dataset.IndexedDatasetBuilder`
and falls back to a raw `MMIDIDX` writer when megatron-core isn't
available.

### Stage 4 — cache
```bash
python scripts/data/prepare_cache_megacpp.py
```
Validates the `.bin`/`.idx` pair via `megatron.core.datasets.indexed_dataset.IndexedDataset`
and reports `docs`, `tokens`, `dtype`. On dev laptops without
megatron-core installed this stage logs a SKIP and exits 0.

Megatron's own `GPTDataset` sample-index lives at
`${MEGACPP_DATA_ROOT}/megatron/cache/GPTDataset_indices/` and is built at
the first training launch — we rely on Megatron to manage it rather than
pre-building here (sample index depends on seed / seq-length / global
batch size and is cheap to regenerate).

### Stage 5 — verify
```bash
python scripts/data/verify_dataset_megacpp.py
```
Checks `.bin`/`.idx` exist + non-empty, parses the index, verifies
`max(token_id) < vocab_size` (default 131072), prints the first 64 tokens
of document 0. Non-zero exit on any failure — **no silent fallbacks**.

---

## Megatron data format

- `.bin` — flat packed token IDs (default dtype `uint32` for vocab=131072).
- `.idx` — `MMIDIDX\x00\x00` magic + version + dtype-code + document count +
  `sizes[int32]` + `pointers[int64]` + `doc_idx[int64]`. Matches the
  `megatron-core` `MMapIndexedDataset` reader.
- On-disk layout:
  ```
  ${MEGACPP_DATA_ROOT}/
  ├── cpp_raw/…                       # stage 1
  ├── parquet/clang_semantic_4k_v10/  # stage 2
  │   ├── shard_00000.parquet
  │   ├── …
  │   ├── val_shard.parquet
  │   └── _COMPLETE
  ├── tokenizer/tokenizer.json        # stage 2 copy
  └── megatron/                       # stage 3
      ├── clang_semantic_4k_v10_train.bin
      ├── clang_semantic_4k_v10_train.idx
      ├── clang_semantic_4k_v10_valid.bin
      ├── clang_semantic_4k_v10_valid.idx
      └── cache/GPTDataset_indices/   # built at first training launch
  ```

---

## How to point training at this data

`scripts/remote_smoke_h200_dsa_9_4_m.sh` (and siblings) hard-code:

```bash
--data-path 1.0 "${REMOTE_ROOT}/data/megatron/clang_semantic_4k_v10_train" \
--tokenizer-type HuggingFaceTokenizer \
--tokenizer-model "${REMOTE_ROOT}/data/tokenizer" \
--split 98,1,1 \
```

so as long as `${REMOTE_ROOT}/data/` matches `${MEGACPP_DATA_ROOT}`
(bench3: `/mnt/data/cppmega-root/data`, europe:
`/home/dave/cppmega-root/data`) the launcher picks it up without change.

To override at launch time without editing the script, patch the
`--data-path` argument in an `EXTRA_FLAGS` wrapper or hand-edit the
launcher block.

---

## Storage requirements

| stage | artifact | approx size |
|---|---|---|
| 1 | raw clones (`cpp_raw/`) | ~15 GB |
| 2 | enriched JSONL (transient) | ~20-30 GB |
| 2 | parquet shards | ~8-12 GB |
| 3 | `.bin` train+valid | ~6-10 GB |
| 3 | `.idx` train+valid | <100 MiB |
| 4 | Megatron sample cache | <1 GiB (built at training launch) |

Ranges depend on how much of each repo survives the "compilable / parseable"
filter inside the clang indexer.

---

## Reproducibility

- **Pinned refs:** every repo URL in `prepare_download_megacpp.sh` carries
  an explicit tag (`llvmorg-19.1.0`, `v6.10`, `boost-1.86.0`, …).
  Shallow clones mean the exact commit is determined by the remote tag —
  there is a small risk of retagging upstream; mirror the raw corpus
  tarball if absolute reproducibility matters.
- **Tokenizer version:** tied to the nanochat commit you point
  `MEGACPP_NANOCHAT_ROOT` at. Record that commit hash together with any
  checkpoint.
- **Deterministic preprocessing:** the streaming parquet writer uses a
  fixed `--seed=42`; the clang indexer is order-sensitive on file-system
  enumeration and is NOT guaranteed bitwise-reproducible across kernels.
  For audits, hash the resulting `.bin` file:
  ```
  sha256sum ${MEGACPP_DATA_ROOT}/megatron/clang_semantic_4k_v10_train.bin
  ```
  and pin that hash in the experiment log.

---

## Honesty notes / gaps

- **Nanochat dep:** stage 2 dispatches into
  `nanochat/scripts/data/run_clang_pipeline.sh` rather than re-implementing
  the clang indexer + tokenizer inside cppmega. This is intentional —
  vendoring ~thousands of LoC of actively-maintained tokenizer code would
  be worse than a path dependency on the upstream checkout. If that
  coupling is unacceptable, the alternative is a clean-room
  re-implementation of the indexer; budget accordingly.
- **No pre-built cache in stage 4:** Megatron's GPTDataset sample index
  depends on `--seed`, `--seq-length`, `--global-batch-size`, and
  `--train-iters`; pre-building it in a separate stage would be fragile
  and Megatron rebuilds it in seconds anyway. Stage 4 only warms the
  `.bin`/`.idx` memmap and validates integrity.
- **Raw corpus not versioned:** `cpp_raw/` is ~15 GB and lives outside
  git. If you need to reproduce a run after upstream repos change, mirror
  `cpp_raw/` to cold storage before discarding.
