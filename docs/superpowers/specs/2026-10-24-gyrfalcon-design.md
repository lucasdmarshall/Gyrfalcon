# Gyrfalcon design

Date: 2026-10-24

Gyrfalcon is a Rust RAG library. The CLI binary is `gyr`. An HTTP server is a thin decode-and-call layer over the library. Generation is an external OpenAI-compatible chat call. The default local embedder and the default local cross-encoder reranker both run on ONNX Runtime (`ort`). BM25 is Tantivy, behind an internal trait. USearch is the HNSW index. A segment directory holds both files.

The project name and the Cargo package name are `gyrfalcon`. There is no trailing dash.

## Decisions

- The public library surface is three calls: `index`, `retrieve`, `answer`.
- The pipeline is fixed: prepare once, BM25 and HNSW, RRF in one arena, optional local rerank, prompt, external LLM.
- The pipeline is fixed so that one `prepare` feeds both stages, the two ranked lists meet in one arena, the API stays three calls, and the benchmark always measures those stages. Virtual-call cost against the LLM is not a reason for this shape.
- `ort` is the only local model runtime in v1. An HTTP embedder is optional. Candle is not a v1 backend.
- Tantivy is the v1 lexical engine. A `LexicalIndex` trait sits in front of it. A Gyrfalcon-owned postings format is not part of v1. It is considered only after a benchmark shows Tantivy is the limiter.
- The Tantivy analyzer is Gyrfalcon's. It is registered on the index as `gyr-text-v1`. Query text is prepared once.
- Default quantization is int8 cosine. Binary is an optional mode.
- C++ source is not part of this repository. C++ arrives inside USearch, ONNX Runtime, and SimSIMD. SimSIMD is called directly only to rescore a binary shortlist.
- Deletes and updates do not rewrite a sealed segment. A collection-level tombstone bitmap is the only delete record. Search drops dead chunk ids inside the USearch predicate and the Tantivy collector, so a dead chunk never takes a top-k slot. The dense list is short only when fewer than `limit` live chunks exist. The lexical list is short only when fewer than `limit` live chunks contain a query term.
- Each segment keeps its own Tantivy index. BM25 scoring uses a collection-wide `Bm25StatisticsProvider`. The totals are the sum of per-segment statistics, dead chunks included, precomputed at seal and at merge. Query does not scan postings to subtract tombstones. A merge rewrites the segments it touches and is what drops dead chunks from the totals.
- Auto-merge is a background tiered policy. It merges only small segments that share a size tier, and it never rebuilds a large segment as part of a write call. A segment with `chunk_count >= 100_000` and a dead ratio above 0.20 is rewritten alone, in the background, from its live chunks. The retrieve bench still reports 1-segment and 50-segment layouts. The ingest bench includes merge time.
- A chunk's `chunk_id` does not change when its segment is merged or rewritten. Tombstones are sets of `chunk_id`. A delete that commits while a merge is building stays dead after the swap.
- After a manifest swap, old segment files are deleted only when the last reader of the retired epoch has released them. A failed delete, including a Windows sharing violation on a memory-mapped file, stays queued and is retried.

## Non-goals for v1

- An owned BM25 postings file.
- Candle, llama.cpp, and hand-written SIMD kernels.
- A PDF or layout decoder. The text-only accuracy run uses the dataset's section text.
- Comparing any Gyrfalcon score with RustyRAG's published 94.5% full-set figure.
- A distributed index, multi-node sharding, or authentication.

## Repository

```
gyrfalcon/
  Cargo.toml                  workspace
  crates/gyrfalcon/           library
  crates/gyr-cli/             binary name: gyr
  bench/adapters/             RustyRAG, Swiftide, LlamaIndex runners
  docs/superpowers/specs/2026-10-24-gyrfalcon-design.md
```

The library crate does not depend on the CLI. The CLI and the HTTP server depend on the library. Benchmark adapters are external runners that speak the report schema below. They are not linked into `libgyrfalcon`.

## Pipeline

Ingest, per document:

1. Split the document into chunks.
2. For each chunk, call `prepare` once. This yields `embed_text` and BM25 tokens.
3. Optionally, before `prepare`, call the external LLM for a contextual prefix. Default is off.
4. Embed `embed_text` with the local ort model, in batches.
5. Append the chunk to the writable segment: Tantivy receives a `PreTokenizedString`, USearch receives the vector under the chunk id.

Query:

1. `prepare(query)` once.
2. Embed `prepared.embed_text`. The embedder does not normalize again.
3. BM25 search and HNSW search run concurrently. Both exclude tombstoned chunk ids while they search. Every BM25 segment is scored with the same collection statistics.
4. Merge the BM25 hits by that score, and merge the HNSW hits by distance. Truncate each merged list to its limit.
5. RRF fuses those two lists in one arena.
6. If rerank is enabled, the local cross-encoder scores the top `rerank_k` hits.
7. `retrieve` returns here.
8. `answer` builds the prompt from the top `context_k` hits and streams an external chat completion.

`answer` calls the same retrieve path. It does not search a second way.

## Text preparation

`prepare(source) -> PreparedText` is the only normalizer.

```
PreparedText {
  embed_text: String,   // NFC, trimmed, original case
  tokens: Vec<Token>,   // analyzer output of embed_text
}
```

`gyr-text-v1` token rules, applied to `embed_text`:

1. Split on the Unicode pattern `[^\p{L}\p{N}]+`. Letters and numbers stay.
2. Unicode-lowercase each piece.
3. Drop empty pieces.
4. Drop a piece longer than 40 Unicode scalar values.
5. No stemming. No stopword list.
6. Token position is the surviving token's index, starting at 0.

`embed_text` keeps the original case because the embedding model was trained on natural text. BM25 tokens are the lowercased pieces. Sharing means one `prepare` call and one source string, not one vocabulary.

With a contextual prefix, `embed_text` is the NFC-trimmed prefix, then `\n\n`, then the NFC-trimmed body. Tokens are computed from that full string.

Tantivy:

- The text field's tokenizer name is `gyr-text-v1`.
- That name is registered with a tokenizer that implements the rules above.
- Ingest writes the field as `PreTokenizedString { text: embed_text, tokens }`, so Tantivy does not tokenize that write a second time.
- `chunk_id` is a `u64` fast field. The collector reads it without loading the stored document.
- A query builds Tantivy `TermQuery` clauses (SHOULD, BM25) from `prepared.tokens`. The query parser does not tokenize the raw query.
- Scoring calls `Searcher::search_with_statistics_provider`. The provider is the published collection stats. The segment's own document frequency is not used.
- A unit test inserts one string through the registered tokenizer as plain text and one string through `prepare` plus `PreTokenizedString`. The postings are identical. The fixture includes English, a Myanmar sentence, `error E0308`, whitespace, and an emoji-only string.

Tantivy BM25 parameters are the engine defaults: `k1 = 1.2`, `b = 0.75`.

## LexicalIndex

```
trait LexicalIndex {
  fn add(chunk_id: u64, prepared: &PreparedText);
  fn search(tokens: &[Token], limit: u32) -> Vec<ScoredId>;
  fn seal();
}
```

v1 has one implementation: `TantivyLexical`. `search` is a read. `add` and `seal` are writer operations. The trait is crate-private. Replacing it with an owned postings index is allowed only after `gyr bench retrieve` and `gyr bench ingest` show Tantivy as the limiter, and that result is written down in the bench report.

### Collection BM25 statistics

A segment-local Tantivy index computes IDF and average document length from that segment alone. Those scores are not comparable with another segment's scores, so a merge would rank a rare term in a small segment above the same term in a large one.

Gyrfalcon keeps one Tantivy index per segment and injects collection-wide statistics. A single Tantivy index for the whole collection would also share IDF, and it is the rejected option. The segment directory is the seal and merge unit: `usearch.index` and `tantivy/` are sealed together and never appended to. A collection-wide Tantivy index would stay writable across calls.

The statistics are Lucene's: each segment records its own totals, the collection total is the sum, and dead chunks stay in that sum until a merge rewrites the segment. Query does not walk postings to compute an exact live `doc_freq`. That scan would sit on the retrieve path.

`TantivyLexical::search` calls `search_with_statistics_provider`. The provider reads only the published collection file. It does not open a segment dictionary.

| Method | Value |
|---|---|
| `total_num_docs` | sum of segment `num_docs`, dead chunks included. Copied into the manifest as `bm25_num_docs`. |
| `total_num_tokens` | sum of segment `num_tokens`, repeats included, dead chunks included. Copied into the manifest as `bm25_num_tokens`. |
| `doc_freq` | sum of per-segment document frequencies for that term, dead chunks included |

Average document length is `bm25_num_tokens / bm25_num_docs`. Tantivy still reads each hit's field norm from its own segment. An empty collection does not divide. It returns no hits.

Each segment directory holds `terms.bin`, a sorted table of `(term, doc_freq)` counted while the segment was written, plus `num_docs` and `num_tokens` in `segment.json`. `num_docs` is the number of chunk records stored in that directory, dead ones included. Seal writes the next collection `bm25/{generation}.bin` as the previous collection map plus this segment's `terms.bin`. That addition touches the new segment's vocabulary only. Compaction of a tier writes the new segment's `terms.bin` from the live chunks it keeps, then writes the next collection map as `published - sum(input terms.bin) + new terms.bin`. Delete and update do not subtract. Update only adds the new segment's map. Dead chunks leave the totals when a merge drops them from the files.

The two scalars are in the manifest so a reader has the collection totals without parsing the term map. The term map stays in `bm25/{generation}.bin` because a vocabulary does not belong in `manifest.json`. Both are written and fsynced before the manifest rename. `bm25_generation` names the map. Search uses that pair and no other count.

A fixture of two segments must show that a shared term's collection `doc_freq` is the sum of the segment frequencies, including a tombstoned chunk, and that this score differs from either segment-local BM25 score. After a merge of the segment that held the dead chunk, that term's collection `doc_freq` drops by one. A delete alone must leave `doc_freq` unchanged.

## Chunks

Product ingest (latency and ingest benches) uses this chunker:

- Recorded spans are byte offsets into the original document.
- Target window is 2048 bytes. Overlap is 256 bytes.
- A cut prefers, in order: a blank line, a single newline, `. ` or `။`, otherwise the last UTF-8 boundary inside the window.
- No cut splits a UTF-8 code point.
- Every byte of the document is covered by at least one span.
- Chunk ids are monotonic `u64` values allocated at write time. They are never reused. They are not content hashes.

The accuracy comparison does not use this chunker. See Benchmarks.

## Segment directory

A collection on disk:

```
{collection}/
  manifest.json
  writer.lock
  tombstones/{generation}.bin
  catalog/{generation}.bin
  bm25/{generation}.bin
  segments/{segment_id}/
    segment.json
    usearch.index
    vectors.bin
    tantivy/              Tantivy index directory
    terms.bin
```

`segment_id` is a zero-padded decimal, allocated by the writer. After seal, `usearch.index`, `vectors.bin`, `terms.bin`, and `tantivy/` never change. USearch opens a sealed file with `view` (memory map). Tantivy opens its directory read-only. `writer.lock` holds the writer pid. A second writer gets `WriterBusy` while that pid is alive. A dead pid is reclaimed on the next writer start.

`segment.json` repeats `model_id`, `dimension`, `quantization`, `analyzer_id`, `chunk_count`, `num_docs`, and `num_tokens`. `num_docs` equals `chunk_count`. If `model_id`, `dimension`, `quantization`, or `analyzer_id` disagree with `manifest.json`, open fails with `SegmentCorrupt` and does not return an index.

A writable segment exists only inside a writer call. It is not in the manifest until `seal`. Search reads only sealed segments named by the manifest the request opened. It does not see unsealed chunks.

`index_documents` seals before it returns. It seals when the open segment reaches 100,000 chunks, and it seals any remainder before returning. Seal writes `segment.json`, `terms.bin`, the Tantivy directory, `usearch.index`, and `vectors.bin`, fsyncs the segment directory, then publishes a manifest. The new catalog generation is the previous live rows plus the rows just sealed. The new BM25 generation is the previous collection map plus this segment's `terms.bin`, and the manifest scalars `bm25_num_docs` and `bm25_num_tokens` grow by that segment's counts. Seal does not change the tombstone generation. The call returns after that rename. It does not wait for a merge.

### Tiered merge

Rebuilding every sealed segment into one HNSW graph would stall a write call for minutes on a large collection. Auto-merge does not do that. It merges a handful of small segments that share a size tier, and it leaves larger segments in place.

A segment's tier is `floor(log2(chunk_count))`, with `chunk_count` taken from `segment.json` (stored records, dead included). `floor(log2(1))` is 0. `segments_per_tier` is 8. `max_merge_chunks` is 100,000.

After `index_documents`, delete, or update publishes its manifest, the process schedules one background merger if none is running. The merger:

1. Under the writer lock, picks the lowest tier that contains at least 8 segments whose 8 smallest `chunk_count` values sum to at most 100,000. The inputs are those 8 segments. A segment with `chunk_count` greater than or equal to 100,000 is never an input. If no tier qualifies, the merger stops.
2. Releases the lock and builds a new segment directory from those 8. Every live chunk is copied under its existing `chunk_id`. The builder skips ids that are dead in the tombstone bitmap copied at step 1. This build does not hold the writer lock.
3. Takes the writer lock again. If any input id is no longer in the manifest, it deletes the new directory and stops. It reads the tombstone bitmap named by the manifest now, not the copy from step 1. Catalog rows whose `chunk_id` is in that bitmap are not retargeted. They stay dead. The swap does not lower `tombstone_generation`. A chunk deleted while the build ran is therefore absent from the live catalog and still present in the bitmap, even if the new files also contain its vector. Search keeps filtering that id. The next BM25 map is `published - sum(input terms.bin) + new terms.bin`, where `published` is the map named by the manifest at this moment, so a segment sealed while the build ran stays in the sum. It fsyncs those files and the new segment, then renames `manifest.json` into place. The rename is the only publish. `epoch` increases by 1. Searchers that already opened the previous manifest keep it.
4. Does not delete the input directories in this step. They stay until the reader epoch that mapped them reaches refcount 0. See Reader epochs. A crash before the rename leaves the previous manifest, and the next open deletes an unreferenced new directory only when no reader has it. A crash after the rename leaves the input directories queued for the reaper.
5. Repeats from step 1 until no tier qualifies.

The tombstone bitmap is not cleared. Other segments may still contain dead ids, and search keeps filtering them. `gyr compact --all` is the manual path that rewrites every segment, drops every dead id, and publishes an empty tombstone bitmap. Auto-merge never calls it. `gyr compact` with no flag runs the same tier selection once, synchronously, and then returns.

`index_documents` returning does not mean the merger is idle. `gyr serve` leaves the merger running. `gyr index` and `gyr bench ingest` wait until the merger is idle before the process exits, so a one-shot process still publishes the merge. A test can pause the merger and observe the seal manifest before the swap.

When the sealed count is greater than 32, `IndexReport` and later `retrieve` responses include `warning: "segment_count"`. The write still succeeds. Thirty-two is the fan-out warning, not a merge-everything trigger. The 50-segment retrieve bench sets auto-merge off and seals the same chunks into 50 segments.

### Dead-ratio rewrite

A segment with `chunk_count >= 100_000` never enters a tier merge, so its dead chunks would stay in the files and in the BM25 sums. After the tier pass finds nothing to do, the same background worker looks for a single segment where `chunk_count >= 100_000` and `dead_in_segment / chunk_count > 0.20`. `dead_in_segment` is the number of that segment's stored chunk ids that appear in the current tombstone bitmap. It rewrites the segment with the highest ratio, one segment per pass.

The rewrite copies each live chunk under the same `chunk_id` and skips dead ids. It publishes with the tier-merge swap formula for one input: BM25 map `published - old terms.bin + new terms.bin`, and manifest scalars recomputed from that map. The new tombstone bitmap is the previous bitmap minus the dead ids that were stored in this segment. Those ids are in no other segment. The swap still waits for reader epochs before deleting the old directory.

`dead_ratio_limit` in the manifest is `0.20`. The comparison is strict: a ratio of exactly 0.20 does not rewrite.

### Reader epochs

A publish stamps `epoch` as one greater than the manifest it replaces. `acquire` copies that epoch, maps its segments, and increments an in-process refcount. `release` decrements it. Retrieve and answer hold the guard until the response is finished.

The reaper deletes a retired epoch's segment directories only when its refcount is 0 and the current manifest does not name them. On Unix, unlinking a file that is still mapped can succeed. Gyrfalcon waits for refcount 0 anyway, so a reader does not lose the path it acquired. On Windows, `DeleteFile` fails with a sharing violation while any mapping exists. A failed delete, including that failure, leaves the path queued. The reaper retries on the next `release` and on the next successful publish. The merge does not fail because a delete failed.

One process owns the collection. The refcount is not a cross-process lock. A test holds `acquire` across a swap, asserts the old segment directory still exists, releases, and asserts a deleter then removes it. A second test injects a deleter that fails once with a busy error and succeeds on the retry.

`vectors.bin` is the quantized payload in ascending chunk-id order. Int8 stores `dimension` bytes per vector. Float16 stores two bytes, float32 four. Binary mode stores the int8 matrix here too. The Hamming graph inside `usearch.index` sets a bit when the corresponding int8 byte is greater than 0. Query maps `vectors.bin` only in binary mode, for the SimSIMD rescore. Int8 query does not map it. Compaction reads it and does not ask USearch to return stored vectors.

USearch parameters for every segment:

| Parameter | Value |
|---|---|
| metric | cosine for `i8`, `f16`, and `f32`. Hamming for `b1`. Rescore stays cosine. |
| quantization | manifest value, default `i8` |
| connectivity `M` | 16 |
| `expansion_add` | 128 |
| `expansion_search` | 64 |
| key | chunk id (`u64`) |

Quantization happens once, and `vectors.bin` stores that result. `f32` stores the model output. `f16` stores the same values narrowed to float16. `i8` L2-normalizes, then scales to `[-127, 127]`. `b1` stores that same int8 matrix in `vectors.bin` and passes bits (`byte > 0`) to USearch, so USearch does not quantize the `f32` vector a second time. Query maps `vectors.bin` only for `b1`, where SimSIMD cosine-rescores the top 100 live hits. `i8`, `f16`, and `f32` map `usearch.index` only and do not call SimSIMD.

## Manifest and open

`manifest.json` fields:

| Field | v1 value |
|---|---|
| `format_version` | 1 |
| `model_id` | `BAAI/bge-small-en-v1.5` unless create overrides it |
| `dimension` | 384 for that default model |
| `quantization` | `i8`, `f16`, `f32`, or `b1` |
| `metric` | `cosine` for the collection. Binary segments still rescore with cosine. |
| `analyzer_id` | `gyr-text-v1` |
| `tombstone_generation` | integer |
| `catalog_generation` | integer |
| `bm25_generation` | integer |
| `bm25_num_docs` | sum of segment `num_docs`, dead included |
| `bm25_num_tokens` | sum of segment `num_tokens`, dead included |
| `segments_per_tier` | 8 |
| `max_merge_chunks` | 100000 |
| `dead_ratio_limit` | 0.20 |
| `epoch` | integer, increases by 1 on every publish |
| `sealed_segments` | array of segment ids |

`Index::create(path, config)` writes this file with `tombstone_generation` 0, `catalog_generation` 0, `bm25_generation` 0, `bm25_num_docs` 0, `bm25_num_tokens` 0, `epoch` 1, and `sealed_segments` empty. It also writes empty generation-0 tombstone, catalog, and BM25 stats files. Every later publish writes `epoch` as one greater than the manifest it replaces. `Index::open(path, expected)` reads the manifest and compares `model_id`, `dimension`, `quantization`, and `analyzer_id` to `expected`.

On any mismatch, open returns `IncompatibleIndex` and does not map segment files, does not load the ONNX model, and does not return a handle.

```
IncompatibleIndex {
  field: ModelId | Dimension | Quantization | AnalyzerId,
  manifest_value: String,
  requested_value: String,
}
```

The three fields required at the open gate are `model_id`, `dimension`, and `quantization`. `analyzer_id` uses the same gate: a different tokenizer would query a different index than the one that was written.

The CLI and the HTTP server always pass the process config as `expected`. There is no open path that skips the comparison. Create uses the process config as the manifest. The default process config is the table above.

The ONNX input width of the loaded model must equal `manifest.dimension`. A disagreement is `IncompatibleIndex` on `Dimension`, with `manifest_value` set from the manifest and `requested_value` set from the model file.

HTTP status for this error is 409. Body:

```json
{
  "error": "incompatible_index",
  "field": "quantization",
  "manifest": "i8",
  "requested": "f32"
}
```

A missing collection directory is `IndexNotFound` (HTTP 404). A lock file whose process is still alive is `WriterBusy` (HTTP 409).

## Tombstones, delete, and update

Sealed segment files stay immutable. Tantivy's own `delete_term` is not used. The only delete record is the collection tombstone bitmap, so USearch and Tantivy cannot disagree about liveness.

Chunk ids in the bitmap are dead. A `chunk_id` is allocated once and is the USearch key, the Tantivy fast field, the catalog key, and the tombstone key. A merge or a dead-ratio rewrite copies a live chunk under that same id. It does not renumber. A dead id stays in the sealed files until a tier merge or a dead-ratio rewrite drops it. Search excludes it by id while it is still stored.

Catalog file `catalog/{catalog_generation}.bin` is a sorted table of live rows: `doc_id`, `chunk_id`, `segment_id`, `byte_start`, `byte_end`. Tombstone file `tombstones/{tombstone_generation}.bin` is a sorted unique `u64` array of dead chunk ids. BM25 file `bm25/{bm25_generation}.bin` is the summed term map, dead chunks included until a merge rewrites them. The three counters advance independently. A publish writes each changed file at `current + 1` for that counter, fsyncs it, then replaces `manifest.json` via write-to-temp and rename on the same filesystem.

Delete document:

1. Read the current catalog rows for `doc_id`. An unknown `doc_id` publishes nothing and returns `deleted_chunks: 0`.
2. Write the next tombstone file as the previous dead ids plus those chunk ids.
3. Write the next catalog file without that document's rows.
4. Publish `manifest.json` with the new tombstone and catalog generations. Leave `bm25_generation`, `bm25_num_docs`, and `bm25_num_tokens` unchanged.

Delete of one chunk removes that one id and leaves the document's other chunks live.

Update document, including an unknown `doc_id` (that case is an insert):

1. Take the writer lock. Chunk and embed the new text into a new segment directory, including `segment.json`. Fsync `usearch.index`, `vectors.bin`, and the Tantivy directory. The new chunk ids are on disk and still unreferenced. Delete holds the same lock. Neither runs beside `index_documents`.
2. Build the next tombstone bitmap by adding the document's previous chunk ids. An insert adds none.
3. Build the next catalog: drop the old rows, add the new rows. Each row's `doc_id` is the caller id. The Tantivy document also stores that string in a `raw` field named `doc_key`.
4. Add the new segment's `terms.bin` to the collection BM25 map and add its `num_docs` and `num_tokens` to the manifest scalars. Do not subtract the old chunks. They leave the totals when a merge drops their segment.
5. Publish one manifest that adds the new segment and points at the new catalog, tombstone, and BM25 generations. Old ids become dead in this same rename, not earlier. The call then schedules the background merger and returns. It does not merge before returning.

A crash before the manifest rename leaves the previous manifest in place. Open deletes segment directories the manifest does not list. The old document stays retrievable. After the rename, the old chunk ids are dead and the new segment is listed.

Query exclusion happens inside each engine. There is no pass that searches first and drops dead ids afterward.

- USearch uses `filtered_search`. The predicate is `|key| !tombstones.contains(key)` and it runs during the graph walk. The requested count is the leg limit.
- Tantivy uses a collector over the `chunk_id` fast field. A dead id is not pushed onto the top-k heap. The collector still walks the matching postings, so the heap holds the best live hits.
- The dense merged list has length `min(limit, live chunk count)`. The lexical merged list has length `min(limit, live chunks whose postings contain at least one query term)`. Either list is shorter than `limit` only for that reason. Dead chunks do not consume slots.
- HNSW may stop before it has filled that count. The segment then repeats `filtered_search` with `expansion_search` 128, 256, 512, and 1024. If the count is still short while the segment has enough live chunks, that segment finishes with an exact scan of its `vectors.bin` that keeps the closest live ids. Tantivy does not need this retry. Its collector already visits every match.
- A test tombstones 90 of 100 chunks. Every chunk contains the query term. Limit 10 must return 10 ids from USearch `filtered_search` and 10 ids from the Tantivy collector. None of those ids are dead.

`gyr compact --all` is separate from the tiered merger:

1. Read every sealed segment. Skip tombstoned chunk ids.
2. Write one new segment with a new Tantivy index, a new USearch file, `vectors.bin`, and `terms.bin`, containing only live chunks. Each live chunk keeps its `chunk_id`. Quantized vectors are copied from the old `vectors.bin` files. `doc_key` and chunk text are copied from the old Tantivy stored fields.
3. Write a new catalog that points every live chunk at the new segment.
4. Write an empty tombstone generation and a BM25 map counted from the live chunks only. Set the manifest scalars from that map.
5. Fsync and rename `manifest.json` with the next `epoch`. Old segment directories are queued for the reader-epoch reaper. They are not deleted in this step.

Until a tier merge, a dead-ratio rewrite, or `--all` drops them, dead vectors still occupy the sealed files and the BM25 sums. The 1,000,000-vector RAM measurement uses a collection with an empty tombstone bitmap. A report that has tombstones must say `tombstone_count`.

Readers `acquire` the epoch at the start of a request and `release` it when the response is finished. A publish does not unmap a request already in flight.

## Models

Default embedder: ort, ONNX weights for `BAAI/bge-small-en-v1.5`, 384 dimensions, cosine. The embedder receives `embed_text` and returns `f32` of that width. USearch quantizes to the manifest dtype on add.

Default reranker: ort, ONNX weights for `Xenova/ms-marco-MiniLM-L-6-v2`. It scores `(query, chunk body)` pairs. The prefix is not part of the rerank text. Rerank can be turned off. The off state is a no-op implementation of the same rerank trait, so the pipeline still has the stage.

Optional HTTP embedder: POST a batch of `embed_text` to an OpenAI-compatible embeddings URL. It must return the manifest dimension. It is off unless configured. The accuracy and latency benches use the local ort embedder.

Contextual prefix is off by default. When on, ingest calls the same external LLM client once per chunk, stores the prefix in a Tantivy stored field, and `prepare` sees prefix plus body. Citations return the body. `include_prefix=true` adds the prefix. The ingest bench records a prefix-off row and a prefix-on row. Prefix-on is allowed to be slow. It is not the headline docs/sec.

The external chat client is HTTP, streamed (SSE), with a connection pool. Time inside that call is outside every local clock. `answer` still reports `time_to_request_ms` as the time from query start until the request bytes are written to the socket.

## Retrieval defaults

| Name | Value |
|---|---|
| lexical limit | 50 |
| dense limit | 50 |
| RRF constant | 60 |
| RRF weight, each leg | 1 |
| rerank_k | 32 |
| context_k | 5 |
| binary rescore width | 100 |

RRF score of a hit is the sum over legs of `weight / (60 + rank)`, rank starting at 1. A hit missing from a leg contributes 0 for that leg.

## Public API

```
index_documents(docs) -> IndexReport
delete_document(doc_id) -> DeleteReport
retrieve(query, RetrieveOptions) -> Hits
answer(query, AnswerOptions) -> stream of AnswerEvent
```

`AnswerEvent` is one of: `time_to_request`, `hit` (the citations), `token`, `done`, `error`.

`RetrieveOptions` can disable rerank. `AnswerOptions` names the chat model id and the endpoint. The library reads those from config. v1 does not embed a provider SDK. The wire format is OpenAI chat completions.

## HTTP

`gyr serve` binds the address given by `--bind`. Each handler parses JSON, calls one library function, and writes the result.

| Method | Path | Library call |
|---|---|---|
| GET | `/health` | none |
| POST | `/v1/documents` | `index_documents` |
| DELETE | `/v1/documents/{doc_id}` | `delete_document` |
| POST | `/v1/retrieve` | `retrieve` |
| POST | `/v1/answer` | `answer`, SSE |

The server does not fuse ranks, tokenize, or open USearch on its own. One process holds one collection.

## CLI

| Command | Behavior |
|---|---|
| `gyr index` | create or open, then `index_documents` from files or stdin |
| `gyr query` | `retrieve` or `answer` for one query |
| `gyr serve` | HTTP layer |
| `gyr compact` | one synchronous tiered merge |
| `gyr compact --all` | rewrite every segment and clear tombstones |
| `gyr bench retrieve` | single-flight and concurrent retrieve clocks |
| `gyr bench ingest` | docs/sec |
| `gyr bench ram` | 1,000,000-vector footprint |
| `gyr bench accuracy` | text-only Open RAG Bench slice |

## Benchmarks

Every report is JSON. It records CPU model, physical cores, `ort` intra-op threads, `model_id`, `dimension`, `quantization`, `analyzer_id`, git revision, and dataset revision when a dataset is used.

`ort` intra-op threads are the same in every retrieve column of a report. The value is `min(4, physical cores)` unless the report records an override. Concurrency is the number of queries in flight. It is not an excuse to change the intra-op pool between columns.

### Retrieve latency

Corpus for this bench: the product chunker over plain text, local ort embedder, tombstone count 0, rerank off for the first table.

Protocol: 200 untimed warmup queries, then 1000 timed queries. The query list is fixed and shuffled with seed 1.

Single-flight column: one query at a time. The next query starts after the previous returns. Record p50 and p99 of retrieve latency.

Concurrent columns: in-flight queries of 4, 16, and 32. Workers share the 1000-query list. Record p50, p99, and achieved QPS. Achieved QPS is `1000 / elapsed_seconds` for that column.

A second report repeats the concurrency columns with rerank on, on the one-segment layout. The headline "retrieve" number is the rerank-off, one-segment, single-flight p50 and p99. The sub-10 ms target refers to that number on the bench machine, and the report must name the machine. A result above 10 ms is a real result. It is not omitted.

The rerank-off protocol also runs on a 50-segment layout of the same chunks. `gyr bench retrieve --segments 1` and `gyr bench retrieve --segments 50` each record single-flight p50 and p99 and the concurrent columns at in-flight 4, 16, and 32. The 50-segment build sets auto-merge off and splits the chunks as evenly as the chunk count allows. The report records `segment_count`.

Every retrieve column also records where HNSW stopped, as a percentage of the 1000 timed queries. Warmup queries are not counted. A query is counted once, at the deepest step any of its segments reached:

| Field | Query stopped after |
|---|---|
| `fallback_pct.64` | the first `filtered_search` at `expansion_search` 64 filled the live count |
| `fallback_pct.128` | a retry at 128 was the deepest step |
| `fallback_pct.256` | a retry at 256 was the deepest step |
| `fallback_pct.512` | a retry at 512 was the deepest step |
| `fallback_pct.1024` | a retry at 1024 was the deepest step |
| `fallback_pct.exact` | the segment scanned `vectors.bin` |

The six percentages sum to 100. The same six fields are on the rerank-on report and on both segment layouts.

Time-to-request is measured on the same query list, single-flight and in-flight 16, rerank off and rerank on, on the one-segment layout. The clock stops when the chat-completion request bytes are written to the socket. A local discard server accepts the bytes and does not run a model, so the clock has no provider time in it. The report says `llm: discarded`.

### Ingest

Headline docs/sec is prefix off, local ort embedder, product chunker, auto-merge on. The fixture is 16 files, each indexed by its own `index_documents` call and each producing 10,000 chunks, so the segments share a tier and eight of them sum to 80,000 chunks. That is under `max_merge_chunks`, so the merger runs. The clock starts at the first byte read and stops when the background merger is idle after the last seal. The report's docs/sec uses the 16 files. It also reports chunks/sec, `seal_wall_seconds`, and `merge_wall_seconds`. A row with auto-merge off records seal-only throughput on the same 16 files and is not the headline. A further row turns contextual prefix on, with auto-merge on, and names the LLM used to write prefixes.

### RAM at 1,000,000 vectors

Synthetic chunks, dimension taken from the manifest (384 at the default model), quantization int8, empty tombstones, model loaded after the index is open.

The report has these byte counts:

- `usearch.index` file bytes, summed across segments
- `vectors.bin` file bytes (on disk; unmapped during an int8 retrieve)
- Tantivy directory bytes
- catalog, tombstone, and BM25 stats file bytes
- process RSS with the ONNX model loaded
- process RSS after the model is dropped and the index remains mapped

The planning figure for the vector payload alone at 384-d int8 is `384 * 1 * 1_000_000` bytes, which is 366 MiB. The report's file bytes are the measurement. The planning figure is not a result.

### Accuracy on the text-only slice

Source dataset: Hugging Face `vectara/open_ragbench`, directory shape `official/pdf/arxiv/`.

Questions: `queries.json` entries whose `source` is exactly `text`. The harness aborts if that count is not 1914. It writes the sorted query UUIDs into the report as `question_id_sha256`. Gold answers come from `answers.json`.

#### Ground-truth chunk id

`qrels.json` maps each query UUID to one object, `{ "doc_id", "section_id" }`. Prepare aborts if a text-only query has any other shape. `section_id` is a zero-based index into that paper's `sections` array. Prepare aborts if it points outside the array.

The relevant chunk id is the string `doc_key` `{doc_id}#{section_id}`. One query has one relevant chunk. Gain is 1 for that `doc_key` and 0 for every other retrieved id.

The accuracy fixture stores one chunk per section. A section is not split, so each `doc_key` is written once. Title and abstract are their own keys and are not relevant. The harness still dedupes, because a retriever can emit the same `doc_key` more than once. Before metrics, and before the five context chunks are chosen, the harness rewrites each system's ordered list in one place: keep the first occurrence of a `doc_key` and drop later copies. Rank is the 1-based position in that rewritten list. Rank is absent when the id is not in the list. Gyrfalcon, RustyRAG, Swiftide, and LlamaIndex all go through this harness step. None of them is trusted to dedupe on its own.

Corpus: every paper JSON under `corpus/`. Each section becomes one chunk with that `doc_key`. The indexed text is that section's `text` field only. Tables and images are not indexed. Title and abstract are additional chunks, `{paper_id}#title` and `{paper_id}#abstract`, and they are not qrel targets. All 1000 papers are indexed, including hard negatives. Adapters must not re-split these chunks and must not parse PDFs. After index, each adapter prints its chunk count. The harness aborts the comparison if the four counts differ.

Each system returns at least 10 ordered `doc_key`s from retrieval, before generation. The harness aborts if any list is shorter than 10 before dedupe. The answer prompt receives the first 5 of the harness-deduped list, in that order.

#### Retrieval table

This table does not call a generator or a judge. For each of the 1914 questions, with `rel` meaning the relevant `doc_key` is at rank `r`:

| Metric | Per question |
|---|---|
| recall@5 | 1 when `r` is present and `r <= 5`, otherwise 0 |
| recall@10 | 1 when `r` is present and `r <= 10`, otherwise 0 |
| MRR | `1 / r` when `r` is present, otherwise 0 |
| nDCG@10 | `1 / log2(r + 1)` when `r` is present and `r <= 10`, otherwise 0 |

nDCG uses that value because one relevant document of gain 1 has an ideal DCG of `1 / log2(2) = 1`. Each reported number is the mean over all 1914 questions.

#### Answer-correctness table

This table is the LLM-judge pass rate. It is not an average of the retrieval metrics. A comparison of this table runs only when all four reports share `generator_id`, `answer_prompt_sha256`, `judge_id`, `question_id_sha256`, and `dataset_revision`. The generation LLM, the answer prompt bytes, and the judge model are the same for Gyrfalcon, RustyRAG, Swiftide, and LlamaIndex.

Pass rate is `passes / 1914`. A judge HTTP error or an unparseable verdict counts in `judge_errors`, is not a pass, and stays in the denominator.

The answer prompt is this text, written unchanged to `bench/answer_prompt.txt`. `{{chunks}}` is the five bodies in retrieve order, each prefixed with its `doc_key` in brackets:

```
Answer the question using only the context. If the context does not contain the answer, say you do not know.

Context:
{{chunks}}

Question: {{question}}
```

The judge prompt is this text, written unchanged to `bench/judge_prompt.txt`:

```
You judge one answer. Reply with a single word, pass or fail.
pass means the system answer contains the factual claim of the gold answer and adds nothing that contradicts it.
fail means it misses that claim or contradicts it.

Question: {{question}}
Gold answer: {{gold}}
System answer: {{system}}
```

The report has two tables and no field for a previously published full-set score. RustyRAG's 94.5% (3,045 questions, Docling, Jina, Cerebras) is a different measurement and is not a column.

Systems in both tables:

- Gyrfalcon
- RustyRAG
- Swiftide
- LlamaIndex

Adapter entry points:

- `bench/adapters/rustyrag/run.sh`
- `bench/adapters/swiftide/run.sh`
- `bench/adapters/llamaindex/run.sh`

Each reads `bench/fixtures/text-only/` produced by `gyr bench accuracy --prepare` and writes `bench/reports/{system}.json` with the retrieval means, the ordered `doc_key`s, the pass count, and the ids above.

## Errors

| Condition | Error | HTTP |
|---|---|---|
| manifest field mismatch, including ONNX width | `IncompatibleIndex` | 409 |
| segment metadata disagrees with manifest | `SegmentCorrupt` | 500 |
| collection path missing | `IndexNotFound` | 404 |
| another live writer | `WriterBusy` | 409 |
| empty query after trim | `EmptyQuery` | 400 |
| ONNX file missing | `ModelUnavailable` | 500 |
| chat endpoint non-2xx | `Llm` | 502 |

`retrieve` does not call the LLM. An LLM failure on `answer` still emits the `hit` events already produced, then an `error` event.

## Tests that lock the spec

- `prepare` tokens equal the registered Tantivy tokenizer on the fixture strings, including Myanmar text.
- Query path calls `prepare` once. A test double embedder and a test double lexical index both observe that single value.
- RRF scores match a hand-computed fixture with constant 60.
- Open with a different `model_id`, `dimension`, `quantization`, or `analyzer_id` returns `IncompatibleIndex` and leaves no mapped files (a guard counter in the segment open function stays 0).
- Delete then retrieve does not return the dead chunk, and the collection BM25 `doc_freq` for a term that existed only in that chunk stays the same. After a tiered merge of the segment that stored it, `doc_freq` drops by one. Update returns only the new chunk ids. A manifest renamed halfway is simulated by stopping before rename: the old document remains retrievable.
- A segment of 100 chunks, 90 of them tombstoned, with the query term present in every chunk, returns 10 live ids for limit 10 from both engines.
- Eight segments of 10 chunks and one segment of 1,000 chunks: after the background merge publishes, the eight are one segment and the 1,000-chunk segment is still in the manifest. A live chunk's `chunk_id` is the same before and after. Pausing the merger shows `index_documents` has already returned with the eight segments sealed. A segment whose `chunk_count` is 100,000 is never a tier-merge input.
- A merge build is paused after it has copied chunk `C`. A delete of `C` commits. The swap then publishes. Retrieve does not return `C`. The tombstone bitmap contains `C`. A survivor chunk keeps its `chunk_id`.
- A segment of 100,000 stored chunks with a dead ratio of 0.21 is rewritten in the background to its live chunks. The same `chunk_id`s survive. The dead ids are removed from the tombstone bitmap and from the BM25 sums. A ratio of 0.20 does not rewrite.
- A reader that acquired the pre-swap epoch still sees the old segment directory after the swap. After release, the reaper deletes it. A deleter that returns a busy error the first time leaves the directory in place until the retry.
- Seal at 100,000 is tested with the threshold overridden to 3.
- Binary mode maps `vectors.bin` and rescores with SimSIMD. Int8 mode does not map `vectors.bin` during retrieve.
- `index_documents` is still durable after a crash of the process that returned: the new chunk ids are in a sealed segment named by the manifest.
- Bench smoke in CI: 20 documents, single-flight and in-flight 4, rerank off. The 1,000,000-vector run and the 1914-question run are explicit commands, not CI defaults.

## Worked example of one retrieve

Collection manifest says `bge-small-en-v1.5`, dimension 384, `i8`, `gyr-text-v1`. The caller opens with that same config. `prepare("Memory safety in Rust")` runs once and yields the embed text with its original case and the tokens `memory`, `safety`, `in`, `rust`. Ort embeds those characters. Each Tantivy segment searches those four terms with the collection `Bm25StatisticsProvider`, and its collector skips dead chunk ids. Each USearch segment runs `filtered_search` with the same tombstone predicate. The lexical lists merge by BM25 score. The dense lists merge by distance. RRF fuses the two merged lists. With rerank off, `retrieve` returns. `answer` writes a chat-completion body that contains the top 5 bodies and then waits on the external model, which is outside the retrieve clock.

## v2 changes (2026-10-24)

Measured on Open RAG Bench text-only (1914 questions, 4-core VM). See
`docs/superpowers/plans/2026-10-24-gyrfalcon-v2.md` and `bench/results/`.

- Two tiers replace the single default model. `lite` (the CLI default) embeds with the static
  Model2Vec model `minishlab/potion-retrieval-32M` (512 dimensions, no runtime, ~1,400
  chunks/s on 4 cores). `full` embeds with `bge-small-en-v1.5` int8 ONNX through `ort`.
  The open gate is unchanged: a collection keeps the model it was created with.
- Default fusion is `Rescore`: every candidate from either leg gets its exact cosine from the
  stored vector, then `(1 - w) * minmax(bm25) + w * minmax(cosine)`. `w` is 0.2 for `lite`
  and 0.3 for `full`. RRF stays available as `--fusion rrf`; with a weak dense leg it ranked
  below BM25 alone (MRR 0.563 vs 0.740), while rescore reached 0.752 (`lite`) and 0.773
  (`full`).
- Rescore reads stored vectors through USearch, so int8 retrieve still leaves `vectors.bin`
  unmapped.
- A viewed USearch index reserves 64 search slots; searches beyond that wait instead of
  failing.

### Owned BM25 index replaces Tantivy (format version 2)

The `LexicalIndex` section allows an owned postings index once a bench shows Tantivy is the
limiter. After the ingest work above, it was: per 10,000-chunk seal on Open RAG Bench sections,
Tantivy took 2.69 s and the separate term count 0.75 s, while USearch took 0.66 s and records
plus static embedding about 1.3 s (`examples/profile_ingest`, best of 3, 4-core VM). Tantivy
indexed about 2M tokens/s on its one writer thread, and more writer threads gave 1.4x for 4x
the memory.

Each segment now holds `lexical/` instead of `tantivy/`:

- `terms.dict`: sorted terms with postings offset, segment doc freq, and the largest tf and
  smallest length norm of the list.
- `postings.bin`: per term, ascending doc ordinals (positions in `ids.bin`) and term
  frequencies, 128-doc blocks bit-packed with skip entries (last doc, max tf, min norm), and
  a vint tail. No positions: no query uses them.
- `norms.bin`: one length-norm byte per doc, Tantivy's and Lucene's encoding.
- `docs.bin`: stored chunk fields in 16 KiB LZ4 blocks.

The build interns borrowed token strings, counts `(term, tf)` per doc, and counting-sorts the
pairs by term, so the postings come out in doc order without copying a token. It also yields
the segment's `terms.bin` map, replacing the separate count. A query scores term at a time into
one accumulator per doc and keeps the best live chunks in a heap; the tombstone test runs only
for a hit that would enter the heap.

Scores are Tantivy's BM25 (`k1 = 1.2`, `b = 0.75`, f32, same norm table). Evidence:

- A unit test builds Tantivy (now a dev-dependency) and the owned index over random
  collections with lengths past the exact-norm limit, injected collection statistics, and
  tombstones: 30,135 compared hits, 85% bit-identical scores, the rest within float rounding.
- On SciFact, NFCorpus, ArguAna, and FiQA test queries (2,677), every top-100 list is
  identical to Tantivy's, and BM25 nDCG@10 is unchanged (0.6611, 0.3079, 0.3048, 0.2373).

Measured effect (best of 3, 4-core VM):

| | Tantivy | owned |
|---|---|---|
| lexical build per seal, ORB sections | 3.45 s (index + term count) | 0.35 s |
| lexical build per seal, FiQA | 1.20 s | 0.12 s |
| end-to-end lite ingest, ORB (85 MB) | 8.94 s | 3.34 s |
| end-to-end lite ingest, FiQA (44 MB) | 7.70 s | 5.87 s |
| BM25 leg mean, SciFact / NFCorpus | 293 / 75 us | 104 / 24 us |
| BM25 leg mean, ArguAna / FiQA (6 segments) | 1,098 / 2,793 us | 525 / 906 us |
| lexical bytes, FiQA | 52.3 MB | 43.9 MB |

`manifest.json` `format_version` is 2. Opening an index with any other version fails with
`IncompatibleIndex` on `format_version`; a version 1 index must be rebuilt. The bench RAM
report key `tantivy_bytes` is now `lexical_bytes`.

### Graph-free dense index replaces USearch (format version 3)

With the owned BM25 index in place, the USearch build was the largest seal cost: 0.5-0.75 s
per 10,000 rows and 4.5 s for FiQA's 57,638 rows on 4 cores, rebuilt from scratch at every
merge. It also lost recall: against an exact int8 scan, HNSW (M 16, ef 128/64) found 0.94 to
0.985 of the true top 50.

Each segment now holds `codes.bin` instead of `usearch.index`: RaBitQ codes (Gao and Long,
SIGMOD 2024) without IVF partitions. A row is dequantized, centered on the segment mean,
rotated by a seeded randomized Hadamard transform (padded to a power of two), and stored as the
sign bits of that residual plus four scalars (correction factor, residual norm, `<c, row>`,
`|row|`). Building is one pass, O(rows), on every core. A query runs a cascade:

1. Hamming distance from the query's residual sign bits to every live row (fixed-width popcount
   kernels, about 3 ns per 512-bit row); a histogram cut keeps `max(80 * limit, 4000)` rows,
   split across segments in proportion to their live rows (twice the share, at least 1000).
2. RaBitQ's asymmetric estimate of the cosine for the survivors, from the query quantized to
   4-bit planes; estimates are comparable across segments.
3. One exact int8 cosine rescore of the best `max(16 * limit, 800)` estimates over the whole
   collection, read from the mapped `vectors.bin` in row order with prefetch.

Kernels are compiled for AVX2 + popcnt, popcnt, and baseline, and chosen once at run time.
Dead rows are skipped inside the scan, so there is no retry ladder: `fallback` is always `E64`
and the `testing_starve_hnsw` hook is gone. Thresholds were chosen on recall against an exact
scan, which uses no relevance labels.

Evidence (`examples/profile_dense`, `bench/results/2026-10-25-cloud-4core-graph-free/`):

- recall@50 against exact: FiQA 0.9958 at limit 50 (HNSW 0.94), SciFact 1.000; with
  `bge-small-en-v1.5` 0.9997 to 1.000 on SciFact and NFCorpus (HNSW 0.963 to 0.985).
  Without centering, bge recall fell below HNSW (0.945 to 0.972).
- BEIR lite dense nDCG@10 rose on SciFact (0.6300 to 0.6323), NFCorpus (0.3018 to 0.3076),
  and FiQA (0.1831 to 0.1887), and stayed at 0.4516 on ArguAna. Fused `rescore:0.6` was
  unchanged or higher on all four.
- Ingest, lite, best of 3: FiQA 5.43 s to 2.38 s, ORB sections 3.25 s to 3.00 s.
- Dense leg on FiQA at limit 50: 0.56 ms on one segment, 0.75 ms on six (HNSW 0.18-0.20 ms).
- `gyr` release binary 16.9 MB to 12.4 MB; USearch (C++) and simsimd are dev-dependencies
  for the comparison only.

Scale: the scan is linear. At about 3 ns per row per core for stage 1, a 1,000,000-row
collection costs a few milliseconds per query on one core. IVF partitions inside large
segments, with the same codes, are the path past that; they are not built yet.

`format_version` is 3. The RAM report key `usearch_bytes` is now `codes_bytes`, and
`vectors.bin` is mapped for search (stage 3 and rescore fusion read it).
