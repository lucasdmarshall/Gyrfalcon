# Gyrfalcon engineering review

Date: 2026-10-24. Machine: 4-core shared cloud VM (Intel Xeon @ 2.10GHz). Lite tier unless
stated. Every number below was measured in this session; estimates are labeled as estimates.

## Verdict

The storage and retrieval core is solid and fast at query time (1–7 ms p50). The weak points
are elsewhere: ingest is mostly single-threaded glue and slow tokenization (about 2.4 MB/s end
to end on 4 cores), the cross-encoder costs about 22 ms per pair on CPU, and no fixed
configuration is right for every dataset. Each of those is fixable with ordinary engineering,
and the measurements say where to start.

## What is already strong

- Crash-safe publish by manifest rename, immutable sealed segments, tombstones, background
  tiered merge with stable chunk ids, reader epochs with a retrying reaper. 90+ tests cover the
  hard cases (crash before rename, delete during merge, held readers, busy deletes).
- Open gate on model, dimension, quantization, and analyzer; it runs before any model loads.
- Collection-wide BM25 statistics over per-segment Tantivy indexes. BM25 nDCG@10 matches
  published BEIR BM25 (SciFact 0.661 vs 0.665).
- Rescore fusion beats RRF and both single legs on every dataset tested.
- Measurement discipline: BEIR and Open RAG Bench harnesses, trace-and-replay that matches live
  nDCG exactly, paired bootstrap intervals, a validation/test protocol.

## Measurements

### Query path (median ms per step, from planner traces)

| Dataset (docs) | prepare | BM25 | embed | dense | exact rescore | fetch/hit |
|---|---|---|---|---|---|---|
| NFCorpus (3.6k) | 0.007 | 0.08 | 0.05 | 0.18 | 0.21 | 0.007 |
| SciFact (5.2k) | 0.014 | 0.49 | 0.11 | 0.28 | 0.28 | 0.011 |
| ArguAna (8.7k, long queries) | 0.083 | 1.47 | 0.64 | 0.26 | 0.24 | 0.005 |
| FiQA (57.6k, 6 segments) | 0.012 | **3.42** | 0.08 | **1.55** | 0.39 | 0.012 |

BM25 dominates on larger collections and long queries. The custom live-id collector scores every
matching document, so Tantivy's block-max WAND pruning never runs. FiQA also fans out over six
segments because six 10k segments never reach the eight-per-tier merge threshold.

### Ingest path (Open RAG Bench sections: 20,840 chunks, 85 MB)

| Stage | Time | Threads |
|---|---|---|
| `prepare` (NFC, split, lowercase) | 3.4 s (25 MB/s) | 1 |
| Static embed (HF `tokenizers` WordPiece + mean) | 12–14 s (6.5 MB/s) | rayon |
| Glue: record building, term counting, maps | ~10 s | 1 |
| Tantivy segment build | 7.7 s | 1 writer thread |
| USearch build | 1.3 s | all cores |
| fsync | 0.02 s | |

Text is tokenized twice (our analyzer, then WordPiece), most stages use one core, and the
stages run one after another.

### Cross-encoder (MiniLM-L6, 256 tokens)

About 22 ms per pair on 4 cores, so reranking the top 8 costs ~180 ms against ~2–6 ms for the
whole retrieve.

### Quality on validation splits (nDCG@10, lite)

| Dataset | BM25 | rescore 0.2 | rescore 0.6 | rescore 0.2 + rerank 8 | rescore 0.4 + rerank 32 |
|---|---|---|---|---|---|
| SciFact | 0.670 | 0.679 | 0.698 | 0.692 | 0.702 |
| NFCorpus | 0.267 | 0.304 | 0.316 | 0.311 | 0.325 |
| ArguAna | 0.395 | 0.453 | 0.465 | 0.479 | 0.479 |
| FiQA | 0.240 | 0.247 | 0.262 | 0.281 | **0.325** |
| Open RAG Bench | 0.807 | **0.819** | 0.780 | 0.756 | 0.730 |

The best dense weight is 0.2 on Open RAG Bench and 0.6 elsewhere. Reranking adds +0.08 on FiQA
and costs −0.06 to −0.09 on Open RAG Bench (long sections cut to 256 tokens, where BM25 is
already near the ceiling). No fixed setting wins everywhere, which is the planner's case.

## What to do, by expected impact

### 1. Ingest pipeline (estimate: 5–10x)

- **One text pass.** Normalize once, then produce both the BM25 tokens and the WordPiece ids
  from the same scan. Replace the HF `tokenizers` call for static models with a purpose-built
  WordPiece (trie, longest-match, ASCII fast path, no per-token allocation). Static embedding
  is then a table lookup per id.
- **`prepare` without regex and allocations**: byte-level ASCII fast path, lowercase into a
  reused buffer, NFC only when a non-ASCII byte appears.
- **Parallel everywhere**: shard chunks across cores for prepare, embedding, and term counting;
  give Tantivy one writer thread per core.
- **Pipeline the stages** (bounded channels: parse → prepare → embed → engines) so I/O,
  tokenization, and index building overlap.
- **Term counting with an interner** instead of `BTreeMap<String, _>` entries per chunk.

### 2. Lexical query (estimate: 2–5x on large collections and long queries)

- Override `Collector::collect_segment` to drive `Weight::for_each_pruning` with the tombstone
  check inside the callback, so block-max WAND skips non-competitive blocks.
- Read-optimized compaction: when idle, merge small segments down to few large ones even if a
  tier has fewer than eight.
- Run both legs inline when the embedder is cheap; spawning a thread costs more than a static
  embed.

### 3. Reranking (estimate: 3–6x per pair, more by skipping)

- int8 cross-encoder (`model_quantized.onnx`), 128-token pairs, batch sorted by length.
- Planner decides per query (P3). On Open RAG Bench the right decision is "do not rerank".

### 4. Quality of the lite tier

- Planner P4: per-query dense weight; the 0.2 vs 0.6 split above is the target to beat.
- bge query instruction prefix on the full tier (MTEB evaluates bge with it).
- A better static model: distill bge-small into a Model2Vec table on the user's own corpus at
  ingest, or ship a larger retrieval-tuned static model.

### 5. Scale and robustness

- Every publish rewrites the full catalog and BM25 map (O(collection) per write; ~45 MB at 1M
  chunks). Move to delta generations with periodic checkpoints.
- The catalog lives in RAM as `String` rows; use an on-disk sorted table with an mmap index.
- HNSW parallel build is non-deterministic (SciFact dense nDCG varies 0.622–0.633 between
  builds); offer a deterministic single-thread build for benchmarks.
- HTTP server: thread per connection, `Connection: close`; add keep-alive and a bounded pool.
- Missing for a framework: metadata filters, several collections per process, Python bindings.

### 6. Document parsing (PDF and others)

Gyrfalcon has no parser today, and for real RAG ingestion parsing is usually the slowest stage.
Recommendation:

1. Do not start with a from-scratch PDF engine. Fonts, CMaps, encodings, and broken files take
   years to get right, and that work does not differentiate Gyrfalcon.
2. Benchmark the candidates on the Open RAG Bench arXiv PDFs (speed, text fidelity against the
   dataset's section text, memory): `pdfium-render` (Chromium's Pdfium, BSD), `pdf_oxide`
   (Rust), `unpdf`, `oxidize-pdf`. Avoid MuPDF/Poppler for licensing (AGPL/GPL).
3. Build Gyrfalcon's own Rust layer on top: page-parallel extraction, reading order,
   header/footer removal, de-hyphenation, heading detection that feeds chunk boundaries.
4. Write a native extraction fast path only if the benchmark shows every candidate is the
   bottleneck; then a text-layer-only reader in Rust is the narrow, achievable scope.

## Suggested order

1. Ingest pipeline (item 1), measured with `examples/profile_ingest` and `gyr bench ingest`.
2. Lexical WAND and compaction (item 2), measured with traces.
3. Planner P3/P4 on the traces already recorded, with the int8/128-token reranker.
4. Head-to-head against other frameworks (M9) once 1–3 land.
5. Parser benchmark and the Rust layout layer (item 6).
6. Scale items (item 5) before claiming million-chunk numbers.

## Progress

Ingest, item 1 (lite tier, best of 3, 4-core VM):

| Step | ORB sections (85 MB) | FiQA (44 MB) |
|---|---|---|
| Start | 24.6 s | 19.5 s |
| Regex-free `prepare`, WordPiece fast path, parallel records and embedding | 8.9 s | 7.7 s |
| Owned BM25 index instead of Tantivy (spec, "Owned BM25 index replaces Tantivy") | 3.3 s | 5.9 s |
| Graph-free dense codes instead of USearch HNSW (spec, "Graph-free dense index") | 3.0 s | 2.4 s |

The lexical build is now about 10x cheaper than Tantivy plus the term count, and the BM25 leg
is 2-3x faster before any pruning, with identical rankings. The dense side no longer builds a
graph: RaBitQ-style codes take one pass, and the cascade finds 0.996-1.000 of the exact top 50
where HNSW found 0.94-0.985. On long chunks (ORB) ingest is now bound by record building, static
embedding, and the lexical build.

Lexical query, item 2: the owned index scores exhaustively today. Its skip entries already
carry the per-block max tf and min norm that block-max pruning needs, so WAND or MaxScore is a
reader change, not a format change.

