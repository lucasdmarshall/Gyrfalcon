<sub>[Docs](../README.md) / Reference</sub>

# Rust API

A guided map of the `gyrfalcon` and `gyrfalcon-ingest` crates: which types matter, how they fit
together, and what to watch for. Item-level documentation is generated from the source:

```sh
cargo doc -p gyrfalcon -p gyrfalcon-ingest --no-deps --open
```

> [!NOTE]
> The crates are not on crates.io yet, so there is no docs.rs page. Once they are published the
> same documentation will be at `https://docs.rs/gyrfalcon`. The version is 0.1.0 and the API
> may change before 1.0. Items marked `#[doc(hidden)]` or described as test hooks are not part of
> the API.

## The shape of a program

```rust
use gyrfalcon::{Document, Filter, RetrieveOptions, Tier};

let mut index = Tier::Lite.open(dir, &model_dir, true)?;          // open or create
index.index_documents(&[Document::body("id", "text")])?;           // write (&mut self)
let hits = index.retrieve("query", RetrieveOptions { rerank: false, limit: 10 })?;  // read (&self)
index.flush()?;                                                    // optional: seal the buffer
```

Reads take `&self` and can run from many threads at once; writes take `&mut self`. To mix them
across threads, put the `Index` in an `RwLock`, which is what `gyr serve` and the Python package
do. One process may hold a collection at a time.

## Opening

| Item | Purpose |
|---|---|
| `Tier::{Lite, Full}` | The two presets. `open(dir, model_dir, create)` creates or opens a collection with the tier's identity, embedder, and fusion |
| `Tier::model_dir()`, `files()`, `model_id()`, `config(model_dir)`, `fusion()`, `embedder(model_dir)` | The pieces `open` uses, for programs that assemble their own |
| `Index::create(path, &IndexConfig)` | Creates a collection; fails with `AlreadyExists` if one is there |
| `Index::open(path, &IndexConfig)` | Opens one, refusing an identity mismatch before mapping any file |
| `Index::open_with_model(path, &config, model_dir)` | Opens and attaches an ONNX embedder |
| `Index::create_in_temp(&config)` | A collection in a temporary directory that lives as long as the index; for tests |
| `IndexConfig { model_id, dimension, quantization, analyzer_id }` | The identity a collection is created with and checked against |
| `Quantization::{I8, F16, F32, B1}` | Stored vector format; `I8` is the default and the only one the tiers use |
| `ANALYZER_ID`, `STATIC_MODEL_ID`, `DEFAULT_MODEL_ID`, `DEFAULT_DIMENSION` | Constants for building an `IndexConfig` |

`Index::open` and `Index::create` attach a model-free hashing embedder. Call `set_embedder`
before writing or searching, or use `Tier::open`, which does it for you.

## Writing

| Item | Purpose |
|---|---|
| `Document { doc_id, text, contextual_prefix, metadata }` | One document. Build with `Document::body(id, text)` and `.with_metadata([(k, v), ...])` |
| `index_documents(&[Document]) -> IndexReport` | Chunk, embed, and write. Durable on return |
| `update_document(Document) -> IndexReport` | Replace every chunk of `doc_id` atomically; inserts an unknown id |
| `delete_document(&str) -> DeleteReport` | Retire every chunk of `doc_id` |
| `delete_chunks(&[u64]) -> DeleteReport` | Retire single chunks |
| `flush() -> IndexReport` | Seal the write buffer into a segment |
| `index_documents_with_vectors(&[(Document, Vec<f32>)])` | Write with vectors you computed; each chunk gets its document's vector |
| `IndexReport { chunk_ids, sealed, epoch, warning }` | What a write did; `warning` is `Some("segment_count")` above 32 segments |
| `DeleteReport { deleted_chunks, epoch }` | What a delete did |

`Document::contextual_prefix` asks the configured chat model for a one-sentence prefix per chunk
before embedding. It is off by default, costs one LLM call per chunk, and needs `set_chat`.

## Reading

| Item | Purpose |
|---|---|
| `retrieve(&str, RetrieveOptions) -> Hits` | Hybrid search over every chunk |
| `retrieve_filtered(&str, RetrieveOptions, Option<&Filter>) -> Hits` | The same over chunks whose document matches the filter |
| `RetrieveOptions { rerank, limit }` | `limit` hits; `rerank` runs the attached reranker on the top 32 |
| `Hits { chunk_ids, hits, epoch, warning, .. }` | The result; `epoch` is the published state the query read |
| `Hit { chunk_id, doc_id, body, prefix, metadata, byte_start, byte_end, score }` | One chunk |
| `Filter::new().eq(k, v).any(k, [v1, v2])` | Every clause must hold; `any` matches any listed value |
| `collection_doc_freq(&str) -> u64` | How many chunks contain a word, after analysis |

> [!IMPORTANT]
> `RetrieveOptions::default()` sets `rerank: true`. With no reranker attached the stage keeps
> the fused order but sets every score to 0. Pass `rerank: false` unless you call
> `set_reranker`.

## Fusion

| Item | Purpose |
|---|---|
| `Fusion::{Rrf, Score { dense_weight }, Rescore { dense_weight }, RescoreMax { dense_weight }}` | The rules described in [Hybrid retrieval](../concepts/hybrid-retrieval.md) |
| `index.fusion` | The rule this index uses; `Tier::open` sets the tier's |
| `index.rrf_weights` | `[lexical, dense]` weights for `Fusion::Rrf` |
| `Fusion::with_dense_weight(w)` | The same rule with another weight |
| `LITE_DENSE_WEIGHT` (0.45), `FULL_RESCORE_DENSE_WEIGHT` (0.65), `RESCORE_DENSE_WEIGHT` (0.4) | The tuned weights |
| `rrf`, `rrf_weighted`, `score_fusion`, `max_fusion` | The fusion functions on plain `(id, score)` lists |

## Models

| Item | Purpose |
|---|---|
| `trait Embedder` | `embed`, `model_id`, `dimension`, optional `batch_bytes`; see [Custom embedders](../guides/custom-embedders.md) |
| `StaticEmbedder::from_dir(dir, model_id)` | A Model2Vec table; `.truncated(n)` for Matryoshka models |
| `OrtEmbedder::from_dir(dir, model_id)` | An ONNX encoder with CLS pooling |
| `HttpEmbedder::new(url, dimension)` | An OpenAI-compatible embeddings endpoint; `.with_model`, `.with_api_key` |
| `HashEmbedder::new(dimension)` | Deterministic feature hashing; for tests |
| `index.set_embedder(Box<dyn Embedder>)` | Attach an embedder; its width is checked on every batch |
| `trait Reranker`, `OrtReranker::from_dir(dir)`, `NoopReranker` | Cross-encoder reranking after fusion |
| `index.set_reranker(Box<dyn Reranker>)` | Attach a reranker |

## Answering

`answer(query, AnswerOptions) -> AnswerStream` retrieves the top `context_k` (default 5) chunks,
builds a prompt, and streams an OpenAI-compatible chat completion. The stream yields
`AnswerEvent::TimeToRequest`, one `Hit` per context chunk, then `Token`s and `Done`, or `Error`.

```rust
use gyrfalcon::{AnswerEvent, AnswerOptions, ChatConfig};

index.set_chat(ChatConfig {
    endpoint: "http://localhost:8080/v1/chat/completions".into(),
    model: "my-model".into(),
    api_key: std::env::var("GYR_CHAT_API_KEY").ok(),
});
for event in index.answer("What is a gyrfalcon?", AnswerOptions { rerank: false, ..Default::default() })? {
    if let AnswerEvent::Token(text) = event {
        print!("{text}");
    }
}
```

## Maintenance

| Item | Purpose |
|---|---|
| `compact() -> bool` | One synchronous round of the merge policy; `true` if it published |
| `compact_all() -> bool` | Rewrite every live chunk into one segment and clear the tombstones |
| `wait_until_merge_idle()` | Block until the background merger is idle |
| `merge_error() -> Option<String>` | The last background merge failure, if any |
| `reap_now()` | Delete retired segment directories no reader holds |
| `manifest() -> Manifest` | The published state |
| `segment_chunk_counts() -> Vec<u64>` | Stored chunks per segment |
| `index.auto_merge`, `index.write_buffer`, `index.seal_every` | Merge on or off, buffer size in chunks (0 disables the buffer), chunks per sealed segment |

## Text utilities

| Item | Purpose |
|---|---|
| `chunk(&str) -> Vec<Span>`, `chunk_with(text, window, overlap)` | The product chunker, as byte spans |
| `prepare(&str) -> PreparedText` | Normalized text and analyzed tokens, as indexing sees them |
| `tokenize`, `analyze` | The analyzer's two stages |

## Errors

All fallible calls return `gyrfalcon::Result<T>`, an alias for `Result<T, gyrfalcon::Error>`.

| Variant | Meaning | `http_status()` | `code()` |
|---|---|---|---|
| `IncompatibleIndex { field, manifest_value, requested_value }` | Identity mismatch on open | 409 | `incompatible_index` |
| `WriterBusy { pid }` | Another live process holds the collection | 409 | `writer_busy` |
| `AlreadyExists(path)` | `create` on an existing collection | 409 | `already_exists` |
| `IndexNotFound(path)` | `open` where there is none | 404 | `index_not_found` |
| `EmptyQuery` | Nothing left of the query after trimming | 400 | `empty_query` |
| `ModelUnavailable(msg)` | A model file or ONNX Runtime is missing | 500 | `model_unavailable` |
| `SegmentCorrupt(msg)` | A segment's files disagree with the manifest | 500 | `segment_corrupt` |
| `Llm(msg)` | The chat endpoint failed | 502 | `llm` |
| `Embed`, `Lexical`, `Vector`, `Io`, `Json` | Lower-level failures | 500 | `internal` |

`IncompatibleField` names the mismatched field: `FormatVersion`, `ModelId`, `Dimension`,
`Quantization`, or `AnalyzerId`.

## `gyrfalcon-ingest`

| Item | Purpose |
|---|---|
| `read_file(&Path) -> Result<Extracted>` | PDF, HTML, or text to text, chosen by extension |
| `Extracted { text, page_starts }` | The text and the byte offset of each PDF page |
| `is_supported(&Path)`, `EXTENSIONS` | What a directory walk should read |
| `html::to_text(&str)` | HTML to text without building a DOM |
| `layout::clean(&[String])` | The PDF layout layer over page texts from any extractor |
| `pdf::pdfium()` | The process-wide PDFium binding; fails with a message naming where it looked |

## Modules for evaluation

`gyrfalcon::bench` (report types, latency percentiles, and the nDCG, recall, and MRR metrics) and `gyrfalcon::planner`
(trace, replay, paired bootstrap, the hash split) are public so the benchmark harness and your
own evaluation code can use them. See [Planner](../concepts/planner.md).

---

<sub>Previous: [Python API](python-api.md) · Next: [Configuration](configuration.md)</sub>
