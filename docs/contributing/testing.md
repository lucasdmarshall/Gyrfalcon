<sub>[Docs](../README.md) / Contributing</sub>

# Testing

Gyrfalcon's tests check behavior that is easy to get wrong silently: crash safety, scores that
must match a reference engine bit for bit, deleted chunks that must never surface, and files that
must not be removed while a reader holds them. This page lists the suites, how to run them, and
how to add to them.

## Running

```sh
cargo test --release                                       # everything that needs no models
cargo test --release -p gyrfalcon --test m4_merge          # one suite
cargo test --release -p gyrfalcon --test m4_merge -- delete   # tests whose name contains "delete"
```

`--release` matters: several tests build segments of 100,000 chunks.

Tests that need a model, ONNX Runtime, or a dataset skip themselves, with a message, when the
variable that points to it is unset. CI runs without them. To run everything:

```sh
eval "$(scripts/fetch-models.sh)"
export GYR_STATIC_MODEL_DIR=$GYR_MODELS/potion-retrieval-32M
cargo test --release
```

| Variable | Enables |
|---|---|
| `GYR_STATIC_MODEL_DIR` | The WordPiece fast path checked token for token against `tokenizers` (`static_tokenizer`), and static-model tests in `m3_ort` |
| `GYR_MODEL_DIR` | Tests with the real bge-small ONNX model |
| `GYR_RERANKER_DIR` | Tests with the real MiniLM cross-encoder |
| `GYR_ORT_FIXTURE` | A round trip through ONNX Runtime with tiny generated models; create them with `python3 crates/gyrfalcon/tests/fixtures/make_ort_fixture.py DIR` |
| `ORT_DYLIB_PATH` | Needed by every ONNX test |

## The suites

| Suite | Tests | Covers |
|---|---|---|
| `m1_seal` | 16 | Creating and sealing; the identity check on open, which must reject a mismatch before mapping any file; crash before the manifest rename |
| `m2_retrieve` | 18 | The query path: one `prepare` per query, collection-wide BM25 statistics, 90 of 100 chunks deleted still returning 10 live hits from both legs |
| `m3_ort` | 11 | Embedders and rerankers: width checks, missing files, rerank off never calling the model, the HTTP embedder |
| `m4_merge` | 14 | Tiered merges, the dead-ratio rewrite, stable chunk ids, a delete during a merge, updates that crash, reader epochs and the retrying reaper |
| `m5_serve` | 14 | The HTTP server and `answer`: every route, atomic `PUT`, prompts byte for byte, streamed tokens, an LLM failure after the hits |
| `m6_bench` | 10 | The benchmark harness: report shapes, merge waiting, metric deduplication |
| `p1_planner` | 6 | Trace and replay: replay equals the live ranking, the bootstrap, the hash split |
| `w1_buffer` | 7 | The write-ahead log and buffer: durability across reopen, torn tails, bit-identical scores before and after sealing |
| `f1_filter` | 4 | Metadata filters across both legs, buffered and sealed chunks, reopen, and merge |
| `static_tokenizer` | 1 | The static embedder's tokenizer against the reference on 6,731 texts |
| unit tests in `src/` | 38 | The analyzer, the WAL format, dense kernels, the HTTP client, BM25 against Tantivy, the PDF layout layer, the HTML reader |
| Python, `crates/gyrfalcon-py/tests` | 7 | The package end to end on the lite tier: writes, search, filters, reopen, the lock, threads |

## Reference implementations

Some components are tested against an established implementation rather than against expected
values written by hand. These are development dependencies only; they never ship.

| Component | Reference | Test |
|---|---|---|
| BM25 index and scoring | Tantivy | random collections with injected statistics and tombstones; every score equal or within float rounding, rankings identical |
| Dense cascade | an exact int8 scan, and USearch HNSW | `examples/profile_dense` reports recall against the exact scan |
| Static tokenizer | Hugging Face `tokenizers` | `static_tokenizer` compares every id |
| Metrics | `pytrec_eval` via `bench/compare/common.py` | the harness reproduces `gyr bench beir` to 1e-15 |

## Writing tests

- **Use the test hooks for failure paths.** `Index` has hooks that fail before the manifest
  rename (`fail_before_rename`), pause a merge after its build (`pause_merge_after_copy`), and fail
  file deletes (`deleter_fail_times`). Crash-safety tests use them rather than timing.
- **Use small thresholds instead of large data.** `seal_every`, `write_buffer`, and similar fields
  can be set low so a test exercises a seal or a merge with a handful of chunks.
- **Use the model-free embedders.** `HashEmbedder` and the recording and counting embedders in
  `gyrfalcon::testing` give deterministic vectors and let a test assert what was embedded and how
  often.
- **Integration tests that need the binary** call `common::gyr_binary()`, which builds `gyr` once
  per test process.
- **Name a test after the behavior it locks**, as the existing ones do:
  `open_rejects_model_id_mismatch_without_mapping`, `update_tombstones_the_old_chunks_in_the_same_rename`.

## Quality checks are not unit tests

nDCG and latency are measured with `gyr bench`, not asserted in `cargo test`: they depend on
models, datasets, and the machine. A change that may affect them is measured as described in
[Contributing](CONTRIBUTING.md#retrieval-quality) and [Reproducing](../benchmarks/reproducing.md).

---

<sub>Previous: [Development setup](development-setup.md)</sub>
