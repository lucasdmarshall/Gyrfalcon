# Gyrfalcon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `gyrfalcon` library and `gyr` CLI described in `docs/superpowers/specs/2026-10-24-gyrfalcon-design.md`: a sealed-segment hybrid index, ort embeddings, external chat, and the benchmark harness.

**Architecture:** One process owns one collection. `index_documents` seals immutable segment directories and publishes `manifest.json` by rename. Retrieve fans out over USearch and Tantivy, scores BM25 with precomputed collection statistics, and fuses with RRF. Tier merges and dead-ratio rewrites run after the write returns. Old files are deleted only after the last reader of that epoch releases them.

**Tech Stack:** Rust 2021, Tantivy (`search_with_statistics_provider`, `PreTokenizedString`), USearch (`filtered_search`, int8 cosine), `ort` for ONNX, `serde_json`, `thiserror`, `tempfile` in tests. No hand-written C++.

## Global Constraints

- Package name `gyrfalcon`. Binary name `gyr`. No trailing dash.
- Public calls are `index_documents`, `delete_document`, `retrieve`, `answer`.
- Analyzer id `gyr-text-v1`. BM25 `k1 = 1.2`, `b = 0.75`. RRF constant `60`, both weights `1`.
- Default model `BAAI/bge-small-en-v1.5`, dimension `384`, quantization `i8`, metric cosine. Reranker `Xenova/ms-marco-MiniLM-L-6-v2`.
- USearch `M = 16`, `expansion_add = 128`, `expansion_search = 64`.
- `segments_per_tier = 8`. `max_merge_chunks = 100000`. `dead_ratio_limit = 0.20` (strictly greater). Fan-out warning at 32 sealed segments.
- Chunk window 2048 bytes, overlap 256. Seal a writable segment at 100,000 chunks and again before `index_documents` returns.
- `chunk_id` is a `u64` allocated once. Merges and rewrites copy it unchanged. Tombstones are sets of `chunk_id`.
- BM25 collection totals include dead chunks until a merge or dead-ratio rewrite drops those records. Query does not scan postings to correct them.
- A write call returns after the seal rename. It does not wait for merge. `gyr index` and `gyr bench ingest` wait until the background worker is idle before exit.
- Delete retired segment files only when that epoch's in-process refcount is 0. Any delete error, including Windows sharing violation (`ERROR_SHARING_VIOLATION`, raw os error 32), stays queued.
- Accuracy fixture is one chunk per section. The harness dedupes `doc_key` before metrics and before the five context chunks. Text-only question count is 1914. Do not compare with RustyRAG's published 94.5% full-set score.
- Headline retrieve is rerank-off, one segment, single-flight p50/p99. Reports include `fallback_pct` for expansion 64, 128, 256, 512, 1024, and exact. Ingest headline waits for the merger.

---

## File map

| File | Responsibility |
|---|---|
| `Cargo.toml` | Workspace |
| `crates/gyrfalcon/src/lib.rs` | Re-exports |
| `crates/gyrfalcon/src/error.rs` | `Error`, `IncompatibleField` |
| `crates/gyrfalcon/src/manifest.rs` | `Manifest`, `IndexConfig`, create/open gate |
| `crates/gyrfalcon/src/prepare.rs` | `prepare`, `gyr-text-v1` tokens |
| `crates/gyrfalcon/src/chunk.rs` | 2048/256 chunker |
| `crates/gyrfalcon/src/segment.rs` | Seal files: `segment.json`, `terms.bin`, `vectors.bin`, catalog rows |
| `crates/gyrfalcon/src/epoch.rs` | In-process refcount and reaper |
| `crates/gyrfalcon/src/index.rs` | `Index`, writer lock, `index_documents` |
| `crates/gyrfalcon/src/lexical.rs` | `LexicalIndex`, Tantivy, statistics provider |
| `crates/gyrfalcon/src/vector.rs` | USearch add, `filtered_search`, expansion fallback |
| `crates/gyrfalcon/src/fuse.rs` | RRF |
| `crates/gyrfalcon/src/retrieve.rs` | Fan-out, merge, fallback counters |
| `crates/gyrfalcon/src/embed.rs` | `Embedder`, `OrtEmbedder` |
| `crates/gyrfalcon/src/rerank.rs` | `Reranker`, ort cross-encoder, no-op |
| `crates/gyrfalcon/src/tombstone.rs` | Delete, update, bitmap |
| `crates/gyrfalcon/src/merge.rs` | Tier pick, dead-ratio pick, background swap |
| `crates/gyrfalcon/src/answer.rs` | Prompt and external chat stream |
| `crates/gyrfalcon/src/bench.rs` | Report structs |
| `crates/gyr-cli/src/main.rs` | `gyr` commands and HTTP |
| `crates/gyrfalcon/tests/m1_seal.rs` | Milestone 1 |
| `crates/gyrfalcon/tests/m2_retrieve.rs` | Milestone 2 |
| `crates/gyrfalcon/tests/m3_ort.rs` | Milestone 3 |
| `crates/gyrfalcon/tests/m4_merge.rs` | Milestone 4 |
| `crates/gyrfalcon/tests/m5_serve.rs` | Milestone 5 |
| `crates/gyrfalcon/tests/m6_bench.rs` | Milestone 6 |
| `bench/answer_prompt.txt` | Exact answer prompt from the spec |
| `bench/judge_prompt.txt` | Exact judge prompt from the spec |
| `bench/adapters/*/run.sh` | External systems, same report schema |

Engine files `usearch.index` and `tantivy/` are written by milestone 2 inside the seal that milestone 1 already publishes. Milestone 1 seals the manifest and the non-engine payload so the open gate and crash rule exist before the engines do.

## Milestone 1 — segment format, seal, manifest

Exit command: `cargo test -p gyrfalcon --test m1_seal`

Must pass: quantization / model / dimension / analyzer mismatch returns `IncompatibleIndex` and the segment-open counter stays 0; create writes epoch 1 and empty generation-0 files; `prepare` matches the token rules including Myanmar text; a chunk span never splits a UTF-8 code point; `index_documents` lists the new chunk ids in the manifest it returns; stopping before the rename leaves the previous manifest.

### Task 1: Workspace, errors, manifest gate

**Files:**
- Create: `Cargo.toml`
- Create: `crates/gyrfalcon/Cargo.toml`
- Create: `crates/gyrfalcon/src/lib.rs`
- Create: `crates/gyrfalcon/src/error.rs`
- Create: `crates/gyrfalcon/src/manifest.rs`
- Test: `crates/gyrfalcon/tests/m1_seal.rs`

**Interfaces:**
- Consumes: nothing
- Produces: `IndexConfig`, `Manifest`, `Quantization`, `Error::IncompatibleIndex`, `Index::create`, `Index::open_counting`

- [ ] **Step 1: Write the failing test**

```rust
use gyrfalcon::{Error, IncompatibleField, Index, IndexConfig, Quantization};

#[test]
fn open_rejects_quantization_mismatch_without_mapping() {
    let dir = tempfile::tempdir().unwrap();
    let config = IndexConfig::test_default();
    Index::create(dir.path(), &config).unwrap();
    let mut other = config.clone();
    other.quantization = Quantization::F32;
    let mut opens = 0u32;
    let err = Index::open_counting(dir.path(), &other, &mut opens).unwrap_err();
    match err {
        Error::IncompatibleIndex { field, manifest_value, requested_value } => {
            assert_eq!(field, IncompatibleField::Quantization);
            assert_eq!(manifest_value, "i8");
            assert_eq!(requested_value, "f32");
        }
        other => panic!("{other:?}"),
    }
    assert_eq!(opens, 0);
}
```

Add the same shape for `model_id`, `dimension`, and `analyzer_id` in the same file. `IndexConfig::test_default` uses `BAAI/bge-small-en-v1.5`, dimension 384, `Quantization::I8`, analyzer `gyr-text-v1`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m1_seal open_rejects_quantization_mismatch_without_mapping -- --exact`
Expected: FAIL to compile, `Index` not defined.

- [ ] **Step 3: Write minimal implementation**

```rust
pub fn open_counting(path: &Path, expected: &IndexConfig, opens: &mut u32) -> Result<Manifest, Error> {
    let manifest = Manifest::read(&path.join("manifest.json"))?;
    if manifest.model_id != expected.model_id {
        return mismatch(IncompatibleField::ModelId, &manifest.model_id, &expected.model_id);
    }
    if manifest.dimension != expected.dimension {
        return mismatch(IncompatibleField::Dimension, &manifest.dimension.to_string(), &expected.dimension.to_string());
    }
    if manifest.quantization != expected.quantization {
        return mismatch(IncompatibleField::Quantization, manifest.quantization.as_str(), expected.quantization.as_str());
    }
    if manifest.analyzer_id != expected.analyzer_id {
        return mismatch(IncompatibleField::AnalyzerId, &manifest.analyzer_id, &expected.analyzer_id);
    }
    *opens += 1;
    Ok(manifest)
}
```

`Index::create` writes `manifest.json` with `epoch: 1`, the zero generations, `bm25_num_docs: 0`, `bm25_num_tokens: 0`, `segments_per_tier: 8`, `max_merge_chunks: 100000`, `dead_ratio_limit: 0.20`, `sealed_segments: []`, and empty `tombstones/0.bin`, `catalog/0.bin`, `bm25/0.bin`. `as_str` for quantization is `i8`, `f16`, `f32`, `b1`. A missing directory returns `Error::IndexNotFound` before any compare.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m1_seal -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/gyrfalcon
git commit -m "Add the manifest open gate."
```

### Task 2: prepare and the chunker

**Files:**
- Create: `crates/gyrfalcon/src/prepare.rs`
- Create: `crates/gyrfalcon/src/chunk.rs`
- Modify: `crates/gyrfalcon/tests/m1_seal.rs`

**Interfaces:**
- Consumes: nothing from Task 1
- Produces: `prepare(source: &str) -> PreparedText`, `chunk(text: &str) -> Vec<Span>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn prepare_lowercases_words_and_keeps_embed_case() {
    let prepared = gyrfalcon::prepare("Memory safety in Rust မြန်မာ E0308");
    assert_eq!(prepared.embed_text, "Memory safety in Rust မြန်မာ E0308");
    assert_eq!(
        prepared.tokens.iter().map(|t| t.text.as_str()).collect::<Vec<_>>(),
        vec!["memory", "safety", "in", "rust", "မြန်မာ", "e0308"]
    );
    let only_emoji = gyrfalcon::prepare("😀");
    assert!(only_emoji.tokens.is_empty());
}

#[test]
fn chunk_covers_bytes_without_splitting_utf8() {
    let text = "aaaa\n\nမြန်မာစာ";
    let spans = gyrfalcon::chunk_with(text, 8, 2);
    let mut covered = vec![false; text.len()];
    for span in &spans {
        assert!(text.is_char_boundary(span.start) && text.is_char_boundary(span.end));
        for i in span.start..span.end {
            covered[i] = true;
        }
    }
    assert!(covered.iter().all(|b| *b));
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m1_seal prepare_lowercases_words_and_keeps_embed_case -- --exact`
Expected: FAIL to compile, `prepare` missing.

- [ ] **Step 3: Write minimal implementation**

`prepare` NFC-normalizes and trims into `embed_text` without changing case. Tokens split `embed_text` on `[^\p{L}\p{N}]+`, Unicode-lowercase each piece, drop empty pieces, and drop a piece whose scalar count is greater than 40. Positions are `0..n`.

`chunk_with(text, window, overlap)` walks byte windows of `window` (production calls it as `chunk_with(text, 2048, 256)`). A cut prefers `\n\n`, then `\n`, then `. ` or `။`, else the last char boundary inside the window. The next window starts `overlap` bytes before the cut, on a char boundary.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m1_seal -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon/src/prepare.rs crates/gyrfalcon/src/chunk.rs crates/gyrfalcon/tests/m1_seal.rs crates/gyrfalcon/src/lib.rs
git commit -m "Add prepare and the byte chunker."
```

### Task 3: Seal a segment and publish

**Files:**
- Create: `crates/gyrfalcon/src/segment.rs`
- Create: `crates/gyrfalcon/src/index.rs`
- Modify: `crates/gyrfalcon/tests/m1_seal.rs`

**Interfaces:**
- Consumes: `Manifest`, `prepare`, `chunk`
- Produces: `Index::index_documents(&mut self, docs: &[Document]) -> Result<IndexReport, Error>`, `Document { doc_id: String, text: String, contextual_prefix: bool }`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn index_documents_publishes_chunk_ids_and_epoch_2() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let report = index.index_documents(&[
        Document { doc_id: "a".into(), text: "alpha beta".into(), contextual_prefix: false },
    ]).unwrap();
    assert_eq!(report.chunk_ids.len(), 1);
    let manifest = Manifest::read(&dir.path().join("manifest.json")).unwrap();
    assert_eq!(manifest.epoch, 2);
    assert_eq!(manifest.sealed_segments.len(), 1);
    assert_eq!(manifest.bm25_num_docs, 1);
    assert!(dir.path().join(format!("segments/{}/terms.bin", manifest.sealed_segments[0])).exists());
    assert!(dir.path().join(format!("segments/{}/vectors.bin", manifest.sealed_segments[0])).exists());
}

#[test]
fn crash_before_rename_keeps_the_previous_manifest() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.fail_before_rename = true;
    assert!(index.index_documents(&[Document { doc_id: "a".into(), text: "alpha".into(), contextual_prefix: false }]).is_err());
    let manifest = Manifest::read(&dir.path().join("manifest.json")).unwrap();
    assert!(manifest.sealed_segments.is_empty());
    assert_eq!(manifest.epoch, 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m1_seal index_documents_publishes_chunk_ids_and_epoch_2 -- --exact`
Expected: FAIL to compile, `index_documents` missing.

- [ ] **Step 3: Write minimal implementation**

Assign chunk ids from a monotonic counter stored in the writer. For each span call `prepare` once. Append int8-sized zero vectors of `dimension` bytes to `vectors.bin` in chunk-id order (milestone 3 replaces the zeros with ort output; the file layout is what milestone 1 locks). Write `terms.bin` as sorted unique tokens from that segment with their document frequencies, and `num_tokens` as the occurrence count. Write `segment.json` with the four identity fields plus `chunk_count` and `num_docs` (equal) and `num_tokens`.

Publish by writing `manifest.json.tmp` and renaming it over `manifest.json` on the same directory. `bm25_num_docs` and `bm25_num_tokens` increase by the new segment. `bm25/{next}.bin` is the previous map plus `terms.bin`. `fail_before_rename` returns an error after the segment directory is fsynced and before the rename. The next open deletes a segment directory the manifest does not list only when no reader epoch holds it. Milestone 1 has no readers, so the unreferenced directory is removed on the following open.

A second `Index::writer` while the pid in `writer.lock` is alive returns `Error::WriterBusy`. A dead pid is overwritten.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m1_seal -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Seal segment files and publish the manifest."
```

## Milestone 2 — retrieve (USearch, Tantivy, RRF)

Exit command: `cargo test -p gyrfalcon --test m2_retrieve`

Must pass: the registered tokenizer and `prepare` yield the same tokens; two segments share one collection `doc_freq` that still counts a tombstoned chunk; RRF of ranks `[1, 2]` and `[2, 1]` matches weight `1/(60+rank)`; a 100-chunk segment with 90 tombstones and a term in every chunk returns 10 live ids from both engines; `fallback_pct` records an exact scan when expansion cannot fill the live count; `retrieve` calls `prepare` once.

### Task 4: Tantivy lexical index and the statistics provider

**Files:**
- Create: `crates/gyrfalcon/src/lexical.rs`
- Modify: `crates/gyrfalcon/src/segment.rs`
- Test: `crates/gyrfalcon/tests/m2_retrieve.rs`

**Interfaces:**
- Consumes: `prepare`, sealed `terms.bin`, `bm25_num_docs`, `bm25_num_tokens`
- Produces: `trait LexicalIndex { fn add(&mut self, chunk_id: u64, prepared: &PreparedText); fn search(&self, tokens: &[Token], limit: u32) -> Result<Vec<ScoredId>>; fn seal(&mut self); }`, `TantivyLexical`

Run: `cargo add --manifest-path crates/gyrfalcon/Cargo.toml tantivy`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn collection_doc_freq_is_the_sum_and_ignores_nothing_dead() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[Document { doc_id: "a".into(), text: "rust safety".into() }]).unwrap();
    index.index_documents(&[Document { doc_id: "b".into(), text: "rust cooking".into() }]).unwrap();
    index.testing_tombstone_chunk(0);
    let hits = index.lexical_search(&gyrfalcon::prepare("rust").tokens, 10).unwrap();
    assert!(hits.iter().any(|hit| hit.chunk_id == 0));
    assert_eq!(index.collection_doc_freq("rust"), 2);
}
```

`testing_tombstone_chunk` writes the bitmap and catalog only. It must not rewrite `bm25_num_docs`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m2_retrieve collection_doc_freq_is_the_sum_and_ignores_nothing_dead -- --exact`
Expected: FAIL to compile, `lexical_search` missing.

- [ ] **Step 3: Write minimal implementation**

Register tokenizer `gyr-text-v1` on each segment index. The text field uses that name. `add` writes a `PreTokenizedString` whose tokens are the `prepare` tokens, and a fast field `chunk_id`. `search` builds SHOULD `TermQuery`s from those tokens and calls `search_with_statistics_provider`. The provider's `total_num_docs` and `total_num_tokens` return the manifest scalars. `doc_freq` reads `bm25/{generation}.bin` and does not open segment postings. The collector drops a fast-field `chunk_id` that is in the tombstone bitmap and still fills `limit` from the remaining matches.

Seal now also writes `tantivy/` before the manifest rename. A unit test in this file indexes one string as plain text through the registered tokenizer and one as `PreTokenizedString`, then asserts the token postings match for `Memory`, `မြန်မာ`, `E0308`, whitespace, and `😀`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m2_retrieve -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Score Tantivy with collection BM25 statistics."
```

### Task 5: USearch filtered search and fallback counters

**Files:**
- Create: `crates/gyrfalcon/src/vector.rs`
- Modify: `crates/gyrfalcon/src/segment.rs`
- Test: `crates/gyrfalcon/tests/m2_retrieve.rs`

**Interfaces:**
- Consumes: `vectors.bin`, tombstone bitmap, chunk ids
- Produces: `VectorIndex::add`, `VectorIndex::search(&self, query: &[f32], limit: u32, live: impl Fn(u64) -> bool) -> Result<VectorHits>`, `FallbackStep` enum `E64, E128, E256, E512, E1024, Exact`

Run: `cargo add --manifest-path crates/gyrfalcon/Cargo.toml usearch`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn filtered_search_returns_ten_live_ids_when_ninety_are_dead() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let mut rows = Vec::new();
    for i in 0..100 {
        let mut vector = vec![0.0; 384];
        vector[0] = if i < 90 { -1.0 } else { 1.0 };
        rows.push((Document { doc_id: format!("d{i}"), text: "shared term".into() }, vector));
    }
    index.index_documents_with_vectors(&rows).unwrap();
    for id in 0..90 {
        index.testing_tombstone_chunk(id);
    }
    let mut query = vec![0.0; 384];
    query[0] = 1.0;
    let hits = index.vector_search(&query, 10).unwrap();
    assert_eq!(hits.ids.len(), 10);
    assert!(hits.ids.iter().all(|id| *id >= 90));
    assert_eq!(hits.deepest, FallbackStep::E64);
}
```

The query vector is 384 floats. `index_documents_with_vectors` quantizes and stores those `f32` values.

Add this test to `m1_seal.rs` as part of Task 3. `seal_every` defaults to 100,000. Four chunks with a threshold of 3 seal one segment of 3 and one remainder segment.

```rust
#[test]
fn seal_splits_when_the_threshold_is_three() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.seal_every = 3;
    let docs: Vec<_> = (0..4).map(|i| Document { doc_id: format!("d{i}"), text: format!("word {i}") }).collect();
    index.index_documents(&docs).unwrap();
    assert_eq!(index.manifest().sealed_segments.len(), 2);
}
```

`seal_every` defaults to 100_000. Four chunks with a threshold of 3 seal one segment of 3 and one remainder segment.

Add this test to `m2_retrieve.rs` in Task 5:

```rust
#[test]
fn binary_mode_maps_vectors_bin_for_rescore() {
    let mut config = IndexConfig::test_default();
    config.quantization = Quantization::B1;
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &config).unwrap();
    let mut vector = vec![0.0; 384];
    vector[0] = 1.0;
    index.index_documents_with_vectors(&[(Document { doc_id: "a".into(), text: "rust".into() }, vector)]).unwrap();
    let mapped = index.vector_search_mapped_paths();
    assert!(mapped.iter().any(|path| path.ends_with("vectors.bin")));
}
```

```rust
#[test]
fn int8_search_does_not_map_vectors_bin() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let mut vector = vec![0.0; 384];
    vector[0] = 1.0;
    index.index_documents_with_vectors(&[(Document { doc_id: "a".into(), text: "rust".into() }, vector.clone())]).unwrap();
    let hits = index.vector_search(&vector, 1).unwrap();
    assert_eq!(hits.ids, vec![0]);
    assert!(index.vector_search_mapped_paths().iter().all(|path| !path.ends_with("vectors.bin")));
}
```

Binary rescore uses SimSIMD cosine on the top 100 live hits. With one hit the rescore list has length 1. Int8 search does not call SimSIMD.

Add this test to `m3_ort.rs` in Task 7:

```rust
#[test]
fn http_embedder_rejects_the_wrong_width() {
    let embedder = HttpEmbedder::new("http://127.0.0.1:9/v1/embeddings", 384);
    let err = embedder.check_width(768).unwrap_err();
    match err {
        Error::IncompatibleIndex { field, .. } => assert_eq!(field, IncompatibleField::Dimension),
        other => panic!("{other:?}"),
    }
}
```

`HttpEmbedder` is off unless configured. `check_width` is what ingest calls after a batch response. It does not open a socket when the configured dimension already disagrees with the manifest. The port-9 URL must not be contacted in this test.

Add this test to `m4_merge.rs` in Task 9:

```rust
#[test]
fn thirty_three_segments_warn() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.auto_merge = false;
    for i in 0..33 {
        index.index_documents(&[Document { doc_id: format!("d{i}"), text: "word".into() }]).unwrap();
    }
    assert_eq!(index.last_report().warning.as_deref(), Some("segment_count"));
}
```

Add this test to `m6_bench.rs` in Task 14:

```rust
#[test]
fn ram_report_lists_the_byte_counts() {
    let report = gyrfalcon::bench_ram_shape();
    for key in ["usearch_bytes", "vectors_bin_bytes", "tantivy_bytes", "catalog_bytes", "tombstone_bytes", "bm25_bytes", "rss_model_loaded", "rss_model_dropped"] {
        assert!(report.get(key).is_some(), "{key}");
    }
}
```

`bench_ram_shape` returns those keys. `gyr bench ram` fills them from a synthetic 1,000,000-vector run and is not the unit test. The unit test locks the key set. `vectors_bin_bytes` stays unmapped during an int8 retrieve; the report still records the file length.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m2_retrieve filtered_search_returns_ten_live_ids_when_ninety_are_dead -- --exact`
Expected: FAIL to compile, `vector_search` missing.

- [ ] **Step 3: Write minimal implementation**

Quantize `f32` once: L2-normalize, scale to `[-127, 127]`, store the bytes in `vectors.bin`, and pass those `i8` values to USearch with cosine, `M = 16`, `expansion_add = 128`, `expansion_search = 64`. `search` calls `filtered_search` with `|key| live(key)`. If the returned live count is below `min(limit, live_count)` retry at 128, 256, 512, and 1024. If still short, scan `vectors.bin` for the closest live ids and record `FallbackStep::Exact`. The first attempt that fills the count is the recorded step. Seal writes `usearch.index` before the rename.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m2_retrieve -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Search USearch with a live-id predicate."
```

### Task 6: RRF retrieve

**Files:**
- Create: `crates/gyrfalcon/src/fuse.rs`
- Create: `crates/gyrfalcon/src/retrieve.rs`
- Test: `crates/gyrfalcon/tests/m2_retrieve.rs`

**Interfaces:**
- Consumes: `LexicalIndex::search`, `VectorIndex::search`, `prepare`, `Embedder` (a test double until milestone 3)
- Produces: `fn rrf(lexical: &[u64], dense: &[u64]) -> Vec<(u64, f32)>`, `Index::retrieve(&self, query: &str, opts: RetrieveOptions) -> Result<Hits, Error>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn rrf_uses_constant_60() {
    let fused = gyrfalcon::rrf(&[10, 11], &[11, 10]);
    let score = |rank: u32| 1.0 / (60.0 + rank as f32);
    assert!((fused[0].1 - (score(1) + score(2))).abs() < 1e-6);
}

#[test]
fn retrieve_prepares_the_query_once() {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(CountingEmbedder::new(384)));
    index.index_documents_with_vectors(&[
        (Document { doc_id: "a".into(), text: "rust safety".into() }, vec![1.0; 384]),
    ]).unwrap();
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(hits.embed_calls, 1);
    assert_eq!(hits.chunk_ids, vec![0]);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m2_retrieve rrf_uses_constant_60 -- --exact`
Expected: FAIL to compile, `rrf` missing.

- [ ] **Step 3: Write minimal implementation**

`rrf` ranks from 1. A hit missing from a leg adds 0. Sort by score descending, then chunk id ascending. `retrieve` calls `prepare` once, embeds `embed_text` once, runs the two legs, merges lexical hits by BM25 score and dense hits by ascending distance, truncates each to `limit`, then fuses. `RetrieveOptions::rerank` false uses a no-op reranker. An empty trimmed query returns `Error::EmptyQuery` before either leg.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m2_retrieve -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Fuse BM25 and HNSW with RRF."
```

## Milestone 3 — ort embed and rerank

Exit command: `cargo test -p gyrfalcon --test m3_ort`

Must pass: an ONNX width that disagrees with `manifest.dimension` is `IncompatibleIndex` on `Dimension` and does not return a handle; the reranker orders a scripted high score first; rerank off does not call the session; embed receives `embed_text` unchanged.

### Task 7: Embedder and reranker sessions

**Files:**
- Create: `crates/gyrfalcon/src/embed.rs`
- Create: `crates/gyrfalcon/src/rerank.rs`
- Test: `crates/gyrfalcon/tests/m3_ort.rs`

**Interfaces:**
- Consumes: `Index::open`, `retrieve`
- Produces: `trait Embedder { fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, Error>; fn model_id(&self) -> &str; fn dimension(&self) -> u32; }`, `trait Reranker { fn rerank(&self, query: &str, bodies: &[String]) -> Result<Vec<f32>, Error>; }`, `OrtEmbedder`, `OrtReranker`, `NoopReranker`

Run: `cargo add --manifest-path crates/gyrfalcon/Cargo.toml ort`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn onnx_width_mismatch_is_dimension() {
    let dir = tempfile::tempdir().unwrap();
    Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    let err = Index::open_with_session(dir.path(), &IndexConfig::test_default(), ScriptedSession { width: 768 }).unwrap_err();
    match err {
        Error::IncompatibleIndex { field, manifest_value, requested_value } => {
            assert_eq!(field, IncompatibleField::Dimension);
            assert_eq!(manifest_value, "384");
            assert_eq!(requested_value, "768");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn rerank_off_does_not_call_the_session() {
    let (index, session) = sealed_one_doc();
    let _ = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert_eq!(session.rerank_calls(), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m3_ort onnx_width_mismatch_is_dimension -- --exact`
Expected: FAIL to compile, `open_with_session` missing.

- [ ] **Step 3: Write minimal implementation**

`ScriptedSession` is the test double of the ort session: `width()` and `run(texts) -> Vec<Vec<f32>>`. `OrtEmbedder` builds that session from the ONNX file. `open` compares `session.width()` to `manifest.dimension` before mapping segments for search. A missing file is `Error::ModelUnavailable`. `OrtReranker` scores `(query, body)` pairs and does not receive the contextual prefix. `NoopReranker` returns equal scores and is what `rerank: false` calls. Production defaults are the two model ids in Global Constraints. `index_documents` embeds with the open session and writes those quantized bytes, replacing the zero placeholder from milestone 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m3_ort -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Plug ort in as the embedder and reranker."
```

## Milestone 4 — tombstone, merge, epoch

Exit command: `cargo test -p gyrfalcon --test m4_merge`

Must pass: delete leaves `doc_freq` unchanged and retrieve hides the chunk; eight 10-chunk segments merge and a 1,000-chunk segment stays, with survivor `chunk_id`s unchanged; a delete that commits while the build is paused stays in the bitmap and is not retrieved after the swap; dead ratio 0.21 on a 100,000-chunk segment rewrites it and drops those ids from the bitmap and the BM25 sums, while 0.20 does not; a held reader keeps the old directory after the swap, and a busy deleter retries.

### Task 8: Delete and update

**Files:**
- Create: `crates/gyrfalcon/src/tombstone.rs`
- Modify: `crates/gyrfalcon/src/index.rs`
- Test: `crates/gyrfalcon/tests/m4_merge.rs`

**Interfaces:**
- Consumes: catalog, tombstone bitmap, BM25 map
- Produces: `Index::delete_document(&mut self, doc_id: &str) -> Result<DeleteReport, Error>`, `Index::update_document(&mut self, doc: Document) -> Result<IndexReport, Error>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn delete_hides_the_chunk_and_keeps_doc_freq() {
    let mut index = two_rust_docs();
    let before = index.collection_doc_freq("rust");
    let report = index.delete_document("a").unwrap();
    assert_eq!(report.deleted_chunks, 1);
    assert_eq!(index.collection_doc_freq("rust"), before);
    let hits = index.retrieve("rust", RetrieveOptions { rerank: false, limit: 5 }).unwrap();
    assert!(!hits.chunk_ids.contains(&0));
}

#[test]
fn unknown_delete_publishes_nothing() {
    let mut index = two_rust_docs();
    let epoch = index.manifest().epoch;
    let report = index.delete_document("missing").unwrap();
    assert_eq!(report.deleted_chunks, 0);
    assert_eq!(index.manifest().epoch, epoch);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m4_merge delete_hides_the_chunk_and_keeps_doc_freq -- --exact`
Expected: FAIL to compile, `delete_document` missing.

- [ ] **Step 3: Write minimal implementation**

`delete_document` looks up catalog rows. Unknown id returns `deleted_chunks: 0` without a publish. Otherwise write the next tombstone file and the next catalog without those rows, and rename the manifest. Do not write a new BM25 file and do not change `bm25_num_docs`. `update_document` of an unknown id is an insert. An update seals a new segment, adds its `terms.bin` to the collection map, tombstones the previous chunk ids in the same rename, and schedules the background worker without waiting. Both hold `writer.lock`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m4_merge delete_hides_the_chunk_and_keeps_doc_freq unknown_delete_publishes_nothing -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Tombstone deletes without rewriting BM25 stats."
```

### Task 9: Tier merge, stable chunk ids, delete during merge

**Files:**
- Create: `crates/gyrfalcon/src/merge.rs`
- Test: `crates/gyrfalcon/tests/m4_merge.rs`

**Interfaces:**
- Consumes: `Index::delete_document`, sealed segments, `writer.lock`
- Produces: `fn select_tier(segments: &[(String, u64)]) -> Option<Vec<String>>`, `Index::pause_merge_after_copy(&mut self)`, `MergeWorker::poll`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn delete_during_merge_stays_dead_and_survivor_ids_do_not_change() {
    let mut index = eight_small_segments();
    let survivor = index.chunk_id_for_doc("keep");
    index.pause_merge_after_copy = true;
    index.index_documents(&[Document { doc_id: "ninth".into(), text: "word ".repeat(10) }]).unwrap();
    index.wait_until_merge_copied();
    let doomed = index.chunk_id_for_doc("doomed");
    index.delete_document("doomed").unwrap();
    index.pause_merge_after_copy = false;
    index.wait_until_merge_idle();
    let hits = index.retrieve("word", RetrieveOptions { rerank: false, limit: 50 }).unwrap();
    assert!(!hits.chunk_ids.contains(&doomed));
    assert!(index.tombstone_contains(doomed));
    assert_eq!(index.chunk_id_for_doc("keep"), survivor);
}

#[test]
fn tier_merge_leaves_the_thousand_chunk_segment() {
    let index = eight_tens_and_one_thousand();
    index.wait_until_merge_idle();
    let sizes = index.segment_chunk_counts();
    assert!(sizes.contains(&1_000));
    assert_eq!(sizes.iter().filter(|n| **n < 1_000).count(), 1);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m4_merge delete_during_merge_stays_dead_and_survivor_ids_do_not_change -- --exact`
Expected: FAIL to compile, `pause_merge_after_copy` missing.

- [ ] **Step 3: Write minimal implementation**

```rust
pub fn select_tier(segments: &[(String, u64)]) -> Option<Vec<String>> {
    let mut by_tier: BTreeMap<u32, Vec<(String, u64)>> = BTreeMap::new();
    for (id, count) in segments {
        if *count >= 100_000 { continue; }
        let tier = count.max(1).ilog2();
        by_tier.entry(tier).or_default().push((id.clone(), *count));
    }
    for group in by_tier.values_mut() {
        group.sort_by_key(|(_, count)| *count);
        if group.len() >= 8 && group.iter().take(8).map(|(_, c)| *c).sum::<u64>() <= 100_000 {
            return Some(group.iter().take(8).map(|(id, _)| id.clone()).collect());
        }
    }
    None
}
```

The worker builds outside the lock, copying each live chunk under the same `chunk_id`. At swap it re-reads the current tombstone bitmap, refuses to mark those ids live, and does not decrease `tombstone_generation`. `pause_merge_after_copy` blocks after the new directory exists and before the swap so the test can delete. The 1,000-chunk segment is a different tier and is not an input.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m4_merge delete_during_merge_stays_dead_and_survivor_ids_do_not_change tier_merge_leaves_the_thousand_chunk_segment -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Merge small tiers without renumbering chunk ids."
```

### Task 10: Dead-ratio rewrite

**Files:**
- Modify: `crates/gyrfalcon/src/merge.rs`
- Test: `crates/gyrfalcon/tests/m4_merge.rs`

**Interfaces:**
- Consumes: `select_tier` returning `None`, tombstone bitmap, `terms.bin`
- Produces: `fn dead_ratio_target(segments: &[(String, u64, u64)]) -> Option<String>`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn dead_ratio_above_20_percent_rewrites_one_large_segment() {
    let mut index = one_segment_with_stored_count(100_000);
    index.testing_tombstone_first_n(21_000);
    let before = index.collection_doc_freq("shared");
    index.wait_until_merge_idle();
    assert_eq!(index.segment_chunk_counts(), vec![79_000]);
    assert_eq!(index.collection_doc_freq("shared"), before - 21_000);
    assert_eq!(index.tombstone_len(), 0);
    let mut exact = one_segment_with_stored_count(100_000);
    exact.testing_tombstone_first_n(20_000);
    exact.wait_until_merge_idle();
    assert_eq!(exact.segment_chunk_counts(), vec![100_000]);
}
```

`one_segment_with_stored_count` writes one sealed segment of that many chunks containing the token `shared`, bypassing the 100,000 seal split by one test-only constructor. Every chunk's token is `shared`, so each dead chunk decreases `doc_freq` by 1 when the rewrite publishes.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m4_merge dead_ratio_above_20_percent_rewrites_one_large_segment -- --exact`
Expected: FAIL, the segment count stays 100,000.

- [ ] **Step 3: Write minimal implementation**

```rust
pub fn dead_ratio_target(segments: &[(String, u64, u64)]) -> Option<String> {
    segments.iter()
        .filter(|(_, stored, dead)| *stored >= 100_000 && *dead as f64 / *stored as f64 > 0.20)
        .max_by(|a, b| (a.2 as f64 / a.1 as f64).total_cmp(&(b.2 as f64 / b.1 as f64)))
        .map(|(id, _, _)| id.clone())
}
```

When `select_tier` returns `None`, the worker rewrites that one segment from live chunks, keeps their `chunk_id`s, sets the BM25 map to `published - old terms.bin + new terms.bin`, and removes this segment's dead ids from the tombstone bitmap. It does not clear ids that belong to other segments.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m4_merge dead_ratio_above_20_percent_rewrites_one_large_segment -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Rewrite a large segment once its dead ratio exceeds 20 percent."
```

### Task 11: Reader epoch reaper

**Files:**
- Create: `crates/gyrfalcon/src/epoch.rs`
- Modify: `crates/gyrfalcon/src/merge.rs`
- Test: `crates/gyrfalcon/tests/m4_merge.rs`

**Interfaces:**
- Consumes: manifest `epoch`, segment directories
- Produces: `struct ReaderGuard`, `Reaper::poll(&mut self, fs: &mut dyn FileDeleter)`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn old_segment_remains_until_the_reader_releases() {
    let mut index = eight_small_segments();
    let guard = index.acquire();
    let old = index.segment_dirs();
    index.wait_until_merge_idle();
    assert!(old[0].exists());
    drop(guard);
    index.reap_now();
    assert!(!old[0].exists());
}

#[test]
fn busy_delete_is_retried() {
    let mut index = eight_small_segments();
    index.deleter_fail_times = 1;
    index.wait_until_merge_idle();
    index.reap_now();
    let old = index.retired_dirs();
    assert!(old[0].exists());
    index.reap_now();
    assert!(!old[0].exists());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m4_merge old_segment_remains_until_the_reader_releases -- --exact`
Expected: FAIL, the old directory is already gone, or `acquire` is missing.

- [ ] **Step 3: Write minimal implementation**

`acquire` increments `refcount[epoch]` and returns a guard. `Drop` decrements and calls `reap_now`. The swap queues the input directories under the retired epoch and does not call `remove_dir_all` itself. `reap_now` deletes a queued directory only when its epoch refcount is 0 and the current manifest does not name it. `FileDeleter::remove` returning any error, including raw os error 32, leaves the path queued. The next `reap_now` tries again. `retrieve` holds the guard for the whole call.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m4_merge -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon
git commit -m "Delete retired segments only after the last reader."
```

## Milestone 5 — answer, server, CLI

Exit command: `cargo test -p gyrfalcon --test m5_serve`

Must pass: `answer` emits `time_to_request` before any token and the discard server records the prompt bytes; HTTP 409 body for a quantization mismatch is the JSON in the spec; `gyr index` on a fixture exits 0 and `gyr query` prints the chunk id.

### Task 12: Answer stream

**Files:**
- Create: `crates/gyrfalcon/src/answer.rs`
- Create: `bench/answer_prompt.txt`
- Test: `crates/gyrfalcon/tests/m5_serve.rs`

**Interfaces:**
- Consumes: `Index::retrieve`
- Produces: `Index::answer(&self, query: &str, opts: AnswerOptions) -> Result<AnswerStream, Error>`, events `TimeToRequest { ms }`, `Hit`, `Token`, `Done`, `Error`

Write `bench/answer_prompt.txt` with the answer prompt text from the spec, byte for byte.

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn time_to_request_stops_when_bytes_hit_the_socket() {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let mut index = one_doc();
    index.set_chat_endpoint(format!("http://{addr}/v1/chat/completions"));
    let mut stream = index.answer("rust", AnswerOptions::default()).unwrap();
    let first = stream.next().unwrap();
    assert!(matches!(first, AnswerEvent::TimeToRequest { ms } if ms >= 0));
}
```

The test accepts one connection, reads until `\r\n\r\n` plus `Content-Length`, and does not write a model response before asserting the event. The event must already be available once the request bytes are written.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m5_serve time_to_request_stops_when_bytes_hit_the_socket -- --exact`
Expected: FAIL to compile, `answer` missing.

- [ ] **Step 3: Write minimal implementation**

`answer` calls `retrieve`, fills the spec prompt with the top `context_k` bodies (`context_k` default 5), writes the OpenAI chat JSON, then emits `TimeToRequest`. Later events parse SSE `data:` lines. A non-2xx response emits the hits already produced, then `AnswerEvent::Error` with `Error::Llm`. Contextual prefix stays off unless `Document` sets `contextual_prefix: true`. Prefix text is not sent to the reranker. When prefix is on, ingest calls this same client once per chunk before `prepare`, and `embed_text` is the prefix, then `\n\n`, then the body.

Add this test to `m5_serve.rs`:

```rust
#[test]
fn prefix_off_embeds_the_body_only() {
    let mut index = Index::create_in_temp(&IndexConfig::test_default()).unwrap();
    index.set_embedder(Box::new(RecordingEmbedder::new(384)));
    index.index_documents(&[Document { doc_id: "a".into(), text: "Body text".into(), contextual_prefix: false }]).unwrap();
    assert_eq!(index.embedder_texts(), vec!["Body text".to_string()]);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m5_serve time_to_request_stops_when_bytes_hit_the_socket -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon bench/answer_prompt.txt
git commit -m "Stream answer events after the prompt is written."
```

### Task 13: HTTP and the gyr binary

**Files:**
- Create: `crates/gyr-cli/Cargo.toml`
- Create: `crates/gyr-cli/src/main.rs`
- Modify: `Cargo.toml`
- Create: `bench/judge_prompt.txt`
- Test: `crates/gyrfalcon/tests/m5_serve.rs`

**Interfaces:**
- Consumes: `index_documents`, `delete_document`, `retrieve`, `answer`
- Produces: `gyr index`, `gyr query`, `gyr serve`, `gyr compact`, `gyr compact --all`

Add `crates/gyr-cli` to the workspace members. The server is `std::net::TcpListener` in the binary, not a new web framework. Routes are `GET /health`, `POST /v1/documents`, `DELETE /v1/documents/{doc_id}`, `POST /v1/retrieve`, `POST /v1/answer` (SSE).

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn mismatch_is_http_409() {
    let server = TestServer::spawn();
    server.create_default_index();
    let response = server.post("/v1/retrieve", r#"{"query":"rust","quantization":"f32"}"#);
    assert_eq!(response.status, 409);
    assert_eq!(response.json["error"], "incompatible_index");
    assert_eq!(response.json["field"], "quantization");
    assert_eq!(response.json["manifest"], "i8");
    assert_eq!(response.json["requested"], "f32");
}
```

The test process config is `i8`. The body field `quantization` is the requested value the handler passes to `open`'s expected config. Handlers only decode JSON and call the library.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m5_serve mismatch_is_http_409 -- --exact`
Expected: FAIL, connection refused.

- [ ] **Step 3: Write minimal implementation**

`gyr serve --bind 127.0.0.1:0` prints the bound port on stdout for the test. `gyr index <dir> <files...>` calls `index_documents` and waits for the merger. `gyr query <dir> <text>` prints chunk ids from `retrieve`. `gyr compact` runs one tier pass. `gyr compact --all` rewrites every segment, keeps live `chunk_id`s, clears the tombstone bitmap, and queues old directories for the reaper. Write `bench/judge_prompt.txt` with the judge prompt from the spec.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m5_serve -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml crates/gyr-cli bench/judge_prompt.txt crates/gyrfalcon/tests/m5_serve.rs
git commit -m "Add the gyr CLI and the thin HTTP layer."
```

## Milestone 6 — bench harness

Exit command: `cargo test -p gyrfalcon --test m6_bench`

Must pass: a scripted run writes `fallback_pct` values that sum to 100 for both `--segments 1` and `--segments 50`; ingest JSON has `merge_wall_seconds` greater than 0 on the 16-file fixture and a seal-only row that is not labeled headline; dedupe keeps the first `doc_key`; recall@5, recall@10, MRR, and nDCG@10 match the hand values for rank 1 and rank 11; the answer table is omitted when `generator_id` differs; the report has no key `published_full_set`.

### Task 14: Retrieve and ingest reports

**Files:**
- Create: `crates/gyrfalcon/src/bench.rs`
- Modify: `crates/gyr-cli/src/main.rs`
- Test: `crates/gyrfalcon/tests/m6_bench.rs`

**Interfaces:**
- Consumes: `FallbackStep`, merger idle flag, `Index::retrieve`
- Produces: `gyr bench retrieve`, `gyr bench ingest`, `RetrieveReport`, `IngestReport`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn fallback_percentages_sum_to_100() {
    let report = gyrfalcon::bench_retrieve_scripted(&[
        FallbackStep::E64, FallbackStep::E128, FallbackStep::Exact, FallbackStep::E64,
    ]);
    let sum = report.fallback_pct.values().sum::<f64>();
    assert!((sum - 100.0).abs() < 1e-6);
    assert!((report.fallback_pct["64"] - 50.0).abs() < 1e-6);
    assert!((report.fallback_pct["exact"] - 25.0).abs() < 1e-6);
}
```

The 1000-query protocol is what `gyr bench retrieve` runs. This unit test checks the percentage function on a short scripted list. A second test in the same file runs `gyr bench retrieve --segments 1` and `--segments 50` against 20 local documents and asserts both JSON files contain the six keys and `segment_count`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m6_bench fallback_percentages_sum_to_100 -- --exact`
Expected: FAIL to compile, `bench_retrieve_scripted` missing.

- [ ] **Step 3: Write minimal implementation**

Count each timed query at its deepest segment step only. Percent is `100.0 * count / timed`. Warmup is 200 and is not counted. Timed is 1000. Concurrency columns are 1, 4, 16, 32. Rerank-on is a second report on the one-segment layout and includes the same six fields. `--segments 50` sets auto-merge off.

`gyr bench ingest` indexes 16 generated files, 10,000 chunks each, one call per file, auto-merge on, and waits until the merger is idle. JSON includes `docs_per_sec`, `chunks_per_sec`, `seal_wall_seconds`, `merge_wall_seconds`, and `headline: true`. The auto-merge-off row sets `headline: false`. The test fixture uses 8 files of 10 chunks so a merge still runs and `merge_wall_seconds > 0`; the 16×10,000 shape is the command's default, not the unit test.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m6_bench fallback_percentages_sum_to_100 -- --exact`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon crates/gyr-cli
git commit -m "Report retrieve fallback rates and ingest merge time."
```

### Task 15: Accuracy tables and adapters

**Files:**
- Modify: `crates/gyrfalcon/src/bench.rs`
- Create: `bench/adapters/rustyrag/run.sh`
- Create: `bench/adapters/swiftide/run.sh`
- Create: `bench/adapters/llamaindex/run.sh`
- Test: `crates/gyrfalcon/tests/m6_bench.rs`

**Interfaces:**
- Consumes: `bench/judge_prompt.txt`, `bench/answer_prompt.txt`
- Produces: `fn dedupe_doc_keys(keys: &[String]) -> Vec<String>`, `fn retrieval_metrics(rank: Option<u32>) -> RetrievalScore`, `gyr bench accuracy`

- [ ] **Step 1: Write the failing test**

```rust
#[test]
fn harness_dedupes_before_metrics() {
    let keys = dedupe_doc_keys(&["a#0".into(), "a#0".into(), "b#1".into()]);
    assert_eq!(keys, vec!["a#0".to_string(), "b#1".to_string()]);
    let at_one = retrieval_metrics(Some(1));
    assert_eq!(at_one.recall_at_5, 1.0);
    assert_eq!(at_one.ndcg_at_10, 1.0);
    let at_eleven = retrieval_metrics(Some(11));
    assert_eq!(at_eleven.recall_at_10, 0.0);
    assert_eq!(at_eleven.mrr, 1.0 / 11.0);
    assert_eq!(at_eleven.ndcg_at_10, 0.0);
}

#[test]
fn answer_table_requires_the_same_generator() {
    let mut reports = four_text_only_reports();
    reports[1].generator_id = "other".into();
    let compared = compare_accuracy(&reports).unwrap();
    assert!(compared.answer.is_none());
    assert!(compared.retrieval.is_some());
    assert!(compared.json.get("published_full_set").is_none());
}
```

`retrieval_metrics` uses the spec formulas: recall@5, recall@10, MRR `1/r`, nDCG@10 `1/log2(r+1)` when `r <= 10`.

- [ ] **Step 2: Run test to verify it fails**

Run: `cargo test -p gyrfalcon --test m6_bench harness_dedupes_before_metrics -- --exact`
Expected: FAIL to compile, `dedupe_doc_keys` missing.

- [ ] **Step 3: Write minimal implementation**

`gyr bench accuracy --prepare` reads `queries.json`, keeps `source == "text"`, and aborts unless the count is 1914. It writes one chunk per section with `doc_key` `{doc_id}#{section_id}` using a zero-based section index, and aborts if a qrel points outside `sections`. The compare step dedupes each system's list before metrics and before choosing five context chunks. The answer table is emitted only when `generator_id`, `answer_prompt_sha256`, `judge_id`, `question_id_sha256`, and `dataset_revision` match. `compare_accuracy` in the unit test builds the four reports in process. Each `bench/adapters/*/run.sh` checks that `BENCH_FIXTURE` is set, prints the fixture chunk count, and execs `$BENCH_ADAPTER_CMD` when that variable is set. When it is unset the script exits 2 and writes no report. The unit test sets `BENCH_ADAPTER_CMD` to a tiny Rust example, `crates/gyrfalcon/examples/fake_adapter.rs`, that copies the fixture chunk count into `bench/reports/{system}.json` using the same schema. The example does not parse PDFs.

- [ ] **Step 4: Run test to verify it passes**

Run: `cargo test -p gyrfalcon --test m6_bench -- --test-threads=1`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add crates/gyrfalcon bench
git commit -m "Score the text-only slice as two tables."
```

## Spec coverage

| Spec rule | Task |
|---|---|
| Manifest mismatch, epoch, empty generation-0 files | Task 1 |
| `gyr-text-v1`, chunker | Task 2 |
| Seal, crash before rename, writer lock | Task 3 |
| Collection BM25 including dead, Tantivy collector | Task 4 |
| USearch predicate, expansion fallback | Task 5 |
| RRF and single `prepare` | Task 6 |
| ort width and rerank off | Task 7 |
| Delete does not change `doc_freq` | Task 8 |
| Tier merge, stable `chunk_id`, delete during merge | Task 9 |
| Dead ratio > 0.20 on a segment of at least 100,000 | Task 10 |
| Reader refcount and busy delete retry | Task 11 |
| Time-to-request | Task 12 |
| HTTP 409 and CLI | Task 13 |
| `fallback_pct`, ingest with merge | Task 14 |
| Dedupe, two accuracy tables, no 94.5% column | Task 15 |

## Test helpers

These functions live in the integration test that first needs them. Later tests import them from `crates/gyrfalcon/tests/common/mod.rs`.

```rust
fn two_rust_docs() -> Index {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.index_documents(&[
        Document { doc_id: "a".into(), text: "rust safety".into() },
        Document { doc_id: "b".into(), text: "rust cooking".into() },
    ]).unwrap();
    index.leak_tempdir(dir);
    index
}

fn eight_small_segments() -> Index {
    let dir = tempfile::tempdir().unwrap();
    let mut index = Index::create(dir.path(), &IndexConfig::test_default()).unwrap();
    index.pause_merge_after_copy = true;
    for i in 0..8 {
        let doc_id = if i == 0 { "keep" } else if i == 1 { "doomed" } else { "" };
        let doc_id = if doc_id.is_empty() { format!("d{i}") } else { doc_id.into() };
        let text = "word ".repeat(10);
        index.index_documents(&[Document { doc_id, text }]).unwrap();
    }
    index.leak_tempdir(dir);
    index
}
```

`one_doc`, `sealed_one_doc`, and `one_segment_with_stored_count` use the same `leak_tempdir` pattern. `one_segment_with_stored_count(n)` calls a test-only `Index::seal_n_chunks(n, "shared")` that writes one segment even when `n` is 100,000. `four_text_only_reports` returns four `AccuracyReport` values with identical `generator_id` `"judge-a"`, `question_id_sha256`, and `dataset_revision`, each with `doc_keys` of length 10. `Index::leak_tempdir` stores the `TempDir` so the collection path stays alive until the index drops. `Document::body(doc_id, text)` sets `contextual_prefix` to false. A test literal written as `Document { doc_id, text }` means that constructor.
