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
- Deletes and updates do not rewrite a sealed segment. A collection-level tombstone bitmap is the only delete record.

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
3. BM25 search from `prepared.tokens`. HNSW search from the query vector. These two run concurrently.
4. Drop tombstoned chunk ids.
5. RRF in one arena. The arena stores only live hits.
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
- A query builds Tantivy `TermQuery` clauses (SHOULD, BM25) from `prepared.tokens`. The query parser does not tokenize the raw query.
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
  segments/{segment_id}/
    segment.json
    usearch.index
    vectors.bin
    tantivy/              Tantivy index directory
```

`segment_id` is a zero-padded decimal, allocated by the writer. After seal, `usearch.index`, `vectors.bin`, and `tantivy/` never change. USearch opens a sealed file with `view` (memory map). Tantivy opens its directory read-only. `writer.lock` holds the writer pid. A second writer gets `WriterBusy` while that pid is alive. A dead pid is reclaimed on the next writer start.

`segment.json` repeats `model_id`, `dimension`, `quantization`, and `analyzer_id`. If any of those disagree with `manifest.json`, open fails with `SegmentCorrupt` and does not return an index.

A writable segment exists only inside a writer call. It is not in the manifest until `seal`. Search reads only sealed segments named by the manifest the request opened. It does not see unsealed chunks.

`index_documents` seals before it returns. It seals when the open segment reaches 100,000 chunks, and it seals any remainder before returning. Seal writes `segment.json`, the Tantivy directory, `usearch.index`, and `vectors.bin`, fsyncs the segment directory, then publishes a manifest. The new catalog generation is the previous live rows plus the rows just sealed. Seal does not change the tombstone generation.

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
| `sealed_segments` | array of segment ids |

`Index::create(path, config)` writes this file with `tombstone_generation` 0, `catalog_generation` 0, and `sealed_segments` empty. It also writes empty generation-0 tombstone and catalog files. `Index::open(path, expected)` reads the manifest and compares `model_id`, `dimension`, `quantization`, and `analyzer_id` to `expected`.

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

Chunk ids in the bitmap are dead. A dead id stays in the sealed files until compaction.

Catalog generation `N` is a sorted table of live rows: `doc_id`, `chunk_id`, `segment_id`, `byte_start`, `byte_end`. Tombstone generation `N` is a sorted unique `u64` array of dead chunk ids. Both files are written completely, then fsynced, then published by replacing `manifest.json` via write-to-temp and rename on the same filesystem.

Delete document:

1. Read the current catalog rows for `doc_id`. An unknown `doc_id` publishes nothing and returns `deleted_chunks: 0`.
2. Write `tombstones/{N+1}.bin` containing the previous dead ids plus those chunk ids.
3. Write `catalog/{N+1}.bin` without that document's rows.
4. Publish `manifest.json` with both generations set to `N+1`.

Delete of one chunk removes that one id and leaves the document's other chunks live.

Update document, including an unknown `doc_id` (that case is an insert):

1. Take the writer lock. Chunk and embed the new text into a new segment directory, including `segment.json`. Fsync `usearch.index`, `vectors.bin`, and the Tantivy directory. The new chunk ids are on disk and still unreferenced. Delete holds the same lock. Neither runs beside `index_documents`.
2. Build the next tombstone bitmap by adding the document's previous chunk ids. An insert adds none.
3. Build the next catalog: drop the old rows, add the new rows. Each row's `doc_id` is the caller id. The Tantivy document also stores that string in a `raw` field named `doc_key`.
4. Publish one manifest that adds the new segment, points at the new catalog, and points at the new tombstone generation. Old ids become dead in this same rename, not earlier.

A crash before the manifest rename leaves the previous manifest in place. Open deletes segment directories the manifest does not list. The old document stays retrievable. After the rename, the old chunk ids are dead and the new segment is listed.

Query filter:

- Load the tombstone generation named by the manifest the request opened.
- Each leg asks for `limit` hits. Hits whose ids are dead are discarded before they enter the RRF arena.
- If a leg has fewer than `limit` live hits and the engine returned a full page, that leg is queried again with a larger page. The page sizes are the ascending unique values among `limit`, `limit * 4`, `limit * 16`, and 1024, each capped at 1024. After the last page, the leg returns what it has.
- RRF therefore never spends a slot on a dead id.

Compaction, `gyr compact`:

1. Read every sealed segment.
2. Skip tombstoned chunk ids.
3. Write one new segment with a new Tantivy index, a new USearch file, and a new `vectors.bin`, containing only live chunks. Quantized vectors are copied from the old `vectors.bin` files. `doc_key` and chunk text are copied from the old Tantivy stored fields.
4. Write a new catalog that points every live chunk at the new segment.
5. Write an empty tombstone generation.
6. Publish the manifest, fsync, then delete the old segment directories and the old generation files.

Until compact, dead vectors still occupy the sealed files. The 1,000,000-vector RAM measurement uses a collection with an empty tombstone bitmap. A report that has tombstones must say `tombstone_count`.

Readers copy the manifest generation at the start of a request and keep those maps until the request ends. A publish does not unmap a request already in flight.

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
| `gyr compact` | compaction |
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

A second report repeats the same four columns with rerank on. The headline "retrieve" number is the rerank-off, single-flight p50 and p99. The sub-10 ms target refers to that number on the bench machine, and the report must name the machine. A result above 10 ms is a real result. It is not omitted.

Time-to-request is measured on the same query list, single-flight and in-flight 16, rerank off and rerank on. The clock stops when the chat-completion request bytes are written to the socket. A local discard server accepts the bytes and does not run a model, so the clock has no provider time in it. The report says `llm: discarded`.

### Ingest

Headline docs/sec is prefix off, local ort embedder, product chunker, from the first byte read to the manifest rename after the final seal. The report includes document count, chunk count, and wall seconds. A second row turns contextual prefix on and names the LLM used to write prefixes.

### RAM at 1,000,000 vectors

Synthetic chunks, dimension taken from the manifest (384 at the default model), quantization int8, empty tombstones, model loaded after the index is open.

The report has these byte counts:

- `usearch.index` file bytes, summed across segments
- `vectors.bin` file bytes (on disk; unmapped during an int8 retrieve)
- Tantivy directory bytes
- catalog plus tombstone file bytes
- process RSS with the ONNX model loaded
- process RSS after the model is dropped and the index remains mapped

The planning figure for the vector payload alone at 384-d int8 is `384 * 1 * 1_000_000` bytes, which is 366 MiB. The report's file bytes are the measurement. The planning figure is not a result.

### Accuracy on the text-only slice

Source dataset: Hugging Face `vectara/open_ragbench`, directory shape `official/pdf/arxiv/`.

Questions: `queries.json` entries whose `source` is exactly `text`. The harness aborts if that count is not 1914. It writes the sorted query UUIDs into the report as `question_id_sha256`. Gold answers come from `answers.json`. Relevance comes from `qrels.json`.

Corpus: every paper JSON under `corpus/`. Each section becomes one chunk. `section_id` in `qrels.json` is a zero-based index into that paper's `sections` array. Prepare aborts if any text-only qrel points outside that array, and no comparison runs until the fixture matches the files on disk. The chunk `doc_key` is `{paper_id}#{section_index}`. The indexed text is that section's `text` field only. Tables and images are not indexed. Title and abstract are additional chunks, `{paper_id}#title` and `{paper_id}#abstract`, and they are not qrel targets. All 1000 papers are indexed, including hard negatives. Adapters must not re-split these chunks and must not parse PDFs. After index, each adapter prints its chunk count. The harness aborts the comparison if the four counts differ.

Systems in the comparison, all four, on this same chunk list and these same 1914 questions:

- Gyrfalcon
- RustyRAG
- Swiftide
- LlamaIndex

Generator model id and judge model id are configuration. A comparison runs only when all four reports share `generator_id`, `judge_id`, `question_id_sha256`, and `dataset_revision`. The judge sees the question, the gold answer, and the system answer, and returns pass or fail with the prompt stored in `bench/judge_prompt.txt` inside the repo. Accuracy is `passes / 1914`. Judged failures (judge HTTP error, unparseable verdict) are counted in `judge_errors` and are not passes. They stay in the denominator.

The same run reports `recall_at_5`: a question scores 1 when any of the top 5 chunks has `doc_key` `{qrels.doc_id}#{qrels.section_id}`.

The judge prompt is this text, written unchanged to `bench/judge_prompt.txt`:

```
You judge one answer. Reply with a single word, pass or fail.
pass means the system answer contains the factual claim of the gold answer and adds nothing that contradicts it.
fail means it misses that claim or contradicts it.

Question: {{question}}
Gold answer: {{gold}}
System answer: {{system}}
```

The report schema has no field for a previously published full-set score. RustyRAG's 94.5% (3,045 questions, Docling, Jina, Cerebras) is a different measurement and is not a column in this comparison.

Adapter entry points:

- `bench/adapters/rustyrag/run.sh`
- `bench/adapters/swiftide/run.sh`
- `bench/adapters/llamaindex/run.sh`

Each reads `bench/fixtures/text-only/` produced by `gyr bench accuracy --prepare` and writes `bench/reports/{system}.json` in the schema above.

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
- Delete then retrieve does not return the dead chunk. Update returns only the new chunk ids. A manifest renamed halfway is simulated by stopping before rename: the old document remains retrievable.
- Seal at 100,000 is tested with the threshold overridden to 3.
- Binary mode maps `vectors.bin` and rescores with SimSIMD. Int8 mode does not map `vectors.bin` during retrieve.
- `index_documents` is still durable after a crash of the process that returned: the new chunk ids are in a sealed segment named by the manifest.
- Bench smoke in CI: 20 documents, single-flight and in-flight 4, rerank off. The 1,000,000-vector run and the 1914-question run are explicit commands, not CI defaults.

## Worked example of one retrieve

Collection manifest says `bge-small-en-v1.5`, dimension 384, `i8`, `gyr-text-v1`. The caller opens with that same config. `prepare("Memory safety in Rust")` runs once and yields the embed text with its original case and the tokens `memory`, `safety`, `in`, `rust`. Ort embeds those characters. Tantivy searches those four terms. USearch searches the 384-d vector as int8 cosine. Dead ids are removed. RRF merges the live lists. With rerank off, `retrieve` returns. `answer` writes a chat-completion body that contains the top 5 bodies and then waits on the external model, which is outside the retrieve clock.
