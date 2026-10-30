<sub>[Docs](../README.md) / Reference</sub>

# Configuration

Every setting Gyrfalcon reads, with its default: command-line flags, environment variables, the
HTTP API, and the limits built into the engine.

- [Command line](#command-line)
- [Environment variables](#environment-variables)
- [HTTP API](#http-api)
- [Engine limits and defaults](#engine-limits-and-defaults)

The Python package takes its few options as arguments (`tier`, `models`, `create`, `warm`); see the
[Python API](python-api.md). It reads `GYR_MODELS`, `XDG_CACHE_HOME`, `ORT_DYLIB_PATH`, and
`GYR_PDFIUM` from the environment.

---

## Command line

```text
gyr index   DIR [PATH...]  [--metadata KEY=VALUE]...                 [model flags]
gyr query   DIR TEXT       [--limit N] [--filter KEY=VALUE]... [--rerank] [--answer]  [model flags]
gyr serve   DIR            [--bind ADDR]                             [model flags]
gyr compact DIR            [--all]                                   [model flags]
gyr bench   retrieve | ingest | beir | trace | replay | embed | ram | accuracy
```

### Commands

| Command | Does |
|---|---|
| `gyr index DIR PATH...` | Creates the collection if needed and indexes each file, and every supported file under each directory. With no `PATH`, indexes standard input as one document with id `stdin`. Waits for merges, then prints a JSON report. A file that fails to read is skipped with a message. |
| `gyr query DIR TEXT` | Prints one line per hit: `chunk_id`, `doc_id`, and score, tab-separated. With `--answer`, streams an answer from the chat endpoint instead. |
| `gyr serve DIR` | Serves the [HTTP API](#http-api), creating the collection if needed. The first line on stdout is the bound address, so `--bind 127.0.0.1:0` picks a free port. |
| `gyr compact DIR` | Runs one round of the merge policy synchronously. `--all` rewrites every live chunk into one segment and drops every deleted chunk. |
| `gyr bench ...` | Benchmarks; each writes a JSON report. See [Reproducing](../benchmarks/reproducing.md). |

| Command flag | Default | Meaning |
|---|---|---|
| `index --metadata KEY=VALUE` | none | Metadata for every document of this run; repeatable |
| `query --limit N` | `10` | Hits to print |
| `query --filter KEY=VALUE` | none | Repeat a key for any of its values, use different keys for all of them |
| `query --rerank` | off | Rerank with the cross-encoder in `--reranker-dir` |
| `query --answer` | off | Stream an answer from `--chat-endpoint` |
| `serve --bind ADDR` | `127.0.0.1:7700` | Address to listen on |
| `compact --all` | off | Full rewrite instead of one merge round |

### Model flags

`index`, `query`, `serve`, `compact`, and the `bench` subcommands that build a collection accept
these. A collection must be opened with the same model settings it was created with.

| Flag | Environment | Default | Meaning |
|---|---|---|---|
| `--tier` | `GYR_TIER` | `lite` | `lite` or `full`; sets the defaults below |
| `--models` | `GYR_MODELS` | `.models` | Directory holding one subdirectory per model |
| `--embedder` | `GYR_EMBEDDER` | the tier's | `static`, `ort`, `http`, or `hash` (feature hashing, for tests) |
| `--model-dir` | `GYR_MODEL_DIR` | the tier's directory under `--models` | Model files for `static` or `ort` |
| `--model-id` | `GYR_MODEL_ID` | the tier's | Model id recorded in and checked against the collection |
| `--dimension` | | the model's width, else 384 | Vector width; below a static model's width it truncates (Matryoshka) |
| `--quantization` | | `i8` | Stored vectors: `i8`, `f16`, `f32`, or `b1`. Only `i8` is benchmarked |
| `--embed-url` | `GYR_EMBED_URL` | | Endpoint for `--embedder http` |
| `--fusion` | | the tier's | `rrf`, `score`, `rescore`, or `rescore-max` |
| `--dense-alpha` | | the tier's weight | Dense weight for `score`, `rescore`, `rescore-max`, in `[0, 1]` |
| `--lexical-weight` | | `1.0` | BM25 weight for `--fusion rrf` |
| `--dense-weight` | | `1.0` | Dense weight for `--fusion rrf` |
| `--warm` | `GYR_WARM` | `background` | When open reads mapped search files: `none`, `background`, or `eager`. `background` returns immediately and reads on a thread; a query before that finishes reads exact rows. `eager` finishes before open returns. `none` does not read ahead. `background` is the default because, on the gp3 profile in `bench/results/2026-10-29-warm-modes`, it opened as fast as `none` (FiQA 169 ms, Quora 364 ms) while `eager` held open for 1.1 s and 5.6 s. The first query stays at the unwarmed latency. A process that stays open reaches the eager query latency while the prefetch runs |
| `--reranker-dir` | `GYR_RERANKER_DIR` | | Cross-encoder directory (`model.onnx`, `tokenizer.json`) |
| `--chat-endpoint` | `GYR_CHAT_ENDPOINT` | | Full chat completions URL, for example `http://localhost:8080/v1/chat/completions` |
| `--chat-model` | `GYR_CHAT_MODEL` | `default` | Model name sent to the chat endpoint |

Tier defaults:

| | `--tier lite` | `--tier full` |
|---|---|---|
| `--embedder` | `static` | `ort` |
| `--model-dir` | `$GYR_MODELS/potion-retrieval-32M` | `$GYR_MODELS/bge-small-en-v1.5-q8` |
| `--model-id` | `minishlab/potion-retrieval-32M` | `Xenova/bge-small-en-v1.5-q8` |
| `--fusion` | `rescore-max` | `rescore` |
| `--dense-alpha` | `0.45` | `0.65` |

When `--fusion` names another rule than the tier's, its weight defaults to 0.45 for
`rescore-max` and 0.4 for `score` and `rescore`.

---

## Environment variables

| Variable | Read by | Meaning |
|---|---|---|
| `GYR_MODELS` | Python, CLI | Root directory of tier models |
| `XDG_CACHE_HOME` | Python | Models default to `$XDG_CACHE_HOME/gyrfalcon/models` when `GYR_MODELS` is unset |
| `ORT_DYLIB_PATH` | engine | Path of the ONNX Runtime library. Python sets it from the `onnxruntime` package when unset |
| `GYR_PDFIUM` | ingest | PDFium library or its directory. Python sets it from the `pypdfium2` package when unset |
| `GYR_ORT_THREADS` | engine | ONNX Runtime intra-op threads; default `min(4, cores)` |
| `GYR_TIER`, `GYR_EMBEDDER`, `GYR_MODEL_DIR`, `GYR_MODEL_ID`, `GYR_EMBED_URL`, `GYR_RERANKER_DIR`, `GYR_WARM` | CLI | Defaults for the flags of the same name |
| `GYR_CHAT_ENDPOINT`, `GYR_CHAT_MODEL` | CLI | Defaults for `--chat-endpoint`, `--chat-model` |
| `GYR_CHAT_API_KEY` | CLI | Bearer token sent to the chat endpoint |
| `GYR_JUDGE_ENDPOINT`, `GYR_JUDGE_MODEL`, `GYR_JUDGE_API_KEY` | `gyr bench accuracy` | The model that grades answers |
| `GYR_GIT_REVISION` | `gyr bench` | Revision recorded in reports when `git` is unavailable |

Variables used only by the test suite are listed in [Testing](../contributing/testing.md).

---

## HTTP API

`gyr serve DIR` exposes one collection. Requests and responses are JSON. The server has no
authentication and listens on `127.0.0.1` by default; put a proxy in front of it before exposing
it to a network.

| Method | Path | Does |
|---|---|---|
| `GET` | `/health` | `{"status": "ok", "warm": "background"}` (`warm` is the mode this process opened with) |
| `POST` | `/v1/documents` | Index documents |
| `PUT` | `/v1/documents/{doc_id}` | Replace a document atomically, or insert it (`doc_id` percent-encoded) |
| `DELETE` | `/v1/documents/{doc_id}` | Delete a document (`doc_id` percent-encoded) |
| `POST` | `/v1/retrieve` | Search |
| `POST` | `/v1/answer` | Search and stream an answer (server-sent events) |

### `POST /v1/documents`

```json
{
  "documents": [
    {"doc_id": "falcon", "text": "The gyrfalcon is the largest falcon.", "metadata": {"topic": "birds", "tags": ["arctic", "raptor"]}}
  ]
}
```

| Field | Required | Meaning |
|---|---|---|
| `documents[].doc_id` | yes | Document id |
| `documents[].text` | yes | Document text |
| `documents[].metadata` | no | Object of strings, numbers, booleans, or arrays of them |
| `documents[].contextual_prefix` | no | `true` asks the chat model for a prefix per chunk; default `false` |

Response: `{"chunk_ids": [0], "sealed": [], "epoch": 1, "warning": null}`.

Posting an existing `doc_id` adds chunks beside the old ones. To replace a document, use `PUT`.

### `PUT /v1/documents/{doc_id}`

```json
{"text": "The gyrfalcon is the largest of all falcons.", "metadata": {"topic": "birds"}}
```

Replaces every chunk and the metadata of `doc_id` in one write: a concurrent search sees the old
version or the new one, never both. An unknown `doc_id` is inserted. The body takes `text`
(required), `metadata`, and `contextual_prefix` as in `POST /v1/documents`, and the response has
the same shape.

### `POST /v1/retrieve`

```json
{"query": "largest falcon", "limit": 5, "rerank": false, "filter": {"topic": "birds"}}
```

| Field | Default | Meaning |
|---|---|---|
| `query` | required | The query text |
| `limit` | `10` | Hits to return |
| `rerank` | `false` | Rerank the top 32 with the server's cross-encoder (`--reranker-dir`); without one it changes nothing |
| `filter` | none | Keys to a value or an array of values, as in Python |
| `include_prefix` | `false` | Include each hit's contextual prefix |

Response:

```json
{
  "chunk_ids": [0],
  "hits": [
    {"chunk_id": 0, "doc_id": "falcon", "body": "The gyrfalcon is the largest falcon.",
     "score": 0.873, "byte_start": 0, "byte_end": 36, "metadata": {"topic": "birds", "tags": ["arctic", "raptor"]}}
  ],
  "epoch": 1,
  "fallback": "64",
  "warning": null
}
```

`fallback` is kept for compatibility with the first format and is always the same value.

### `POST /v1/answer`

Request: `{"query": "..."}`, with optional `rerank` (default `false`) and `include_prefix`. The
response is `text/event-stream`:

| Event | Data |
|---|---|
| `time_to_request` | `{"ms": 3.1}`: time until the chat request was written |
| `hit` | one per context chunk, as in `/v1/retrieve` |
| `token` | `{"text": "..."}` |
| `done` | `{}` |
| `error` | an error body; hits already sent remain valid |

The server needs `--chat-endpoint`.

### Identity fields

Any request body may carry `model_id`, `dimension`, `quantization`, or `analyzer_id`. The server
checks them against the collection before doing anything and answers 409 on a mismatch, so a
client built for one model cannot write into a collection of another.

### Errors

```json
{"error": "writer_busy", "message": "another writer (pid 8269) holds the collection"}
{"error": "incompatible_index", "field": "model_id", "manifest": "minishlab/potion-retrieval-32M", "requested": "other"}
```

| Status | `error` |
|---|---|
| 400 | `bad_request`, `empty_query` |
| 404 | `index_not_found`, `not_found` (unknown path) |
| 409 | `incompatible_index`, `writer_busy`, `already_exists` |
| 500 | `model_unavailable`, `segment_corrupt`, `internal` |
| 502 | `llm` |

The server handles each connection on its own thread and closes it after one response.

---

## Engine limits and defaults

Built-in values. None are configurable from Python; Rust programs can change the ones marked
with a field.

| Setting | Value | Rust field |
|---|---|---|
| Chunk window | 2,048 bytes | `chunk_with` for custom splits |
| Chunk overlap | 256 bytes | |
| Candidates per leg before fusion | 50, or the requested limit if larger | |
| Reranked candidates | 32 | |
| Context chunks for `answer` | 5 | `AnswerOptions::context_k` |
| Writes logged and buffered | fewer than 256 chunks | |
| Buffer seals at | 4,096 chunks or 32 MiB | `Index::write_buffer` (0 disables the buffer) |
| Segment size within one write | 100,000 chunks | `Index::seal_every` |
| Background merge | 8 segments of one size class, at most 100,000 chunks | `Index::auto_merge` |
| Large segment rewrite | more than 20% deleted | |
| Segment count warning | above 32 | |
| Ingest embedding batch | 32 texts, 32 KB of text (8 KB for ONNX models) | `Embedder::batch_bytes` |
| Tokens per text | 512 | |
| Dense prefilter keeps | `max(80 * limit, 4000)` rows | |
| Dense exact rescore | `max(16 * limit, 800)` rows | |
| BM25 | `k1` 1.2, `b` 0.75 | |
| RRF constant | 60 | |
| ONNX Runtime threads | `min(4, cores)` | `GYR_ORT_THREADS` |
| Ingest threads | every core in the process's CPU affinity | |
| Python `add_many` batch | 512 documents | |

---

<sub>Previous: [Rust API](rust-api.md) · Next: [Benchmark results](../benchmarks/results.md)</sub>
