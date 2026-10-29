# Changelog

## 0.1.0

Prepared on 2026-10-29. Not published: no `v0.1.0` tag has been pushed, and nothing has been uploaded to PyPI or crates.io.

The Python distribution name is `gyrfalcon-rag` because `gyrfalcon` is already registered on PyPI. The import name is `gyrfalcon`. The Rust crates `gyrfalcon`, `gyrfalcon-ingest`, and `gyr-cli` are version 0.1.0.

### Added

- Hybrid search in one engine: an owned BM25 index (stemming, stopwords, collection-wide statistics) and a graph-free dense cascade, fused by rescoring every candidate with its exact cosine.
- Two tiers. `lite` embeds with potion-retrieval-32M and needs no model runtime. `full` embeds with int8 bge-small-en-v1.5 through ONNX Runtime, loaded at run time.
- Durable writes: a write-ahead log, sealed segments, background merges, and a single atomic rename per state change. An update's new chunks and the removal of its old ones commit together.
- Metadata filters applied inside both legs.
- Optional cross-encoder rerank of the fused head.
- Ingestion of PDF (PDFium plus a layout layer), HTML, Markdown, and plain text.
- The same on-disk collection opens from Rust, from Python, from the `gyr` command, and from the HTTP server.
- Python extras `pdf` (PDFium via `pypdfium2`) and `full` (ONNX Runtime via `onnxruntime`).
- Wheels, on a `v*` tag, for Linux x86_64 and aarch64, macOS arm64 (`macos-14`) and x86_64 (`macos-15-intel`), and Windows x64. One abi3 wheel per platform covers Python 3.9 and later.

The public API is 0.1 and can change before 1.0.
