# Gyrfalcon documentation

Gyrfalcon is a retrieval engine for RAG. It runs BM25 and dense vector search over the same
chunks and fuses them, in one Rust library with a Python package, a command line, and an HTTP
server. These pages cover installing it, using it, how it works inside, and how its benchmark
numbers were produced.

## Where to start

| You want to | Read |
|---|---|
| Search your documents from Python in five minutes | [Python quickstart](getting-started/quickstart-python.md) |
| Embed the engine in a Rust program | [Rust quickstart](getting-started/quickstart-rust.md) |
| Decide between the `lite` and `full` tiers | [Choosing a tier](guides/choosing-a-tier.md) |
| Replace a LangChain or LlamaIndex retriever | [From LangChain](guides/migrating-from-langchain.md), [From LlamaIndex](guides/migrating-from-llamaindex.md) |
| Understand why the engine is built the way it is | [Architecture](concepts/architecture.md) |
| Check the benchmark claims before you trust them | [Methodology](benchmarks/methodology.md), then [Results](benchmarks/results.md) |
| Send a patch | [Contributing](contributing/CONTRIBUTING.md) |

## Contents

### Getting started

- [Installation](getting-started/installation.md): the Python package, the Rust crate, the
  `gyr` binary, models, and the optional ONNX Runtime and PDFium libraries.
- [Python quickstart](getting-started/quickstart-python.md): index a folder and run the first
  query.
- [Rust quickstart](getting-started/quickstart-rust.md): open a collection with a tier, write,
  search, filter.

### Guides

Task-oriented pages. Each one answers "how do I do X".

- [Choosing a tier](guides/choosing-a-tier.md)
- [Indexing large corpora](guides/indexing-large-corpora.md)
- [Custom embedders](guides/custom-embedders.md)
- [Persistence](guides/persistence.md)
- [Migrating from LangChain](guides/migrating-from-langchain.md)
- [Migrating from LlamaIndex](guides/migrating-from-llamaindex.md)

### Concepts

Explanations. Each one answers "why does it work this way".

- [Architecture](concepts/architecture.md): the write path, the read path, and the files on disk.
- [Hybrid retrieval](concepts/hybrid-retrieval.md): the BM25 leg, the dense leg, and fusion.
- [Tiers](concepts/tiers.md): what a tier fixes, and the identity check on open.
- [Planner](concepts/planner.md): trace and replay, and how every default was chosen.

### Reference

- [Python API](reference/python-api.md)
- [Rust API](reference/rust-api.md)
- [Configuration](reference/configuration.md): every flag, environment variable, HTTP field,
  and built-in limit, with its default.

### Benchmarks

- [Results](benchmarks/results.md): quality, speed, and memory against LlamaIndex, LangChain,
  and Haystack, with run-to-run variance.
- [Methodology](benchmarks/methodology.md): datasets, the tuning and held-out split, what each
  number includes, and what is disclosed.
- [Reproducing](benchmarks/reproducing.md): the commands that regenerate every table.

### Design records

Decisions that shaped the engine, with the evidence behind them.

| Record | Decision |
|---|---|
| [0001](design/0001-hybrid-fusion.md) | Fuse by rescoring every candidate with its exact cosine, not by rank |
| [0002](design/0002-tier-system.md) | Two named tiers instead of free model configuration |
| [0003](design/0003-query-term-frequency.md) | A query term repeated n times weighs n in BM25 |

The full engineering specification, with every format change and measurement since the first
version, is [`superpowers/specs/2026-10-24-gyrfalcon-design.md`](superpowers/specs/2026-10-24-gyrfalcon-design.md).
Features designed for a later version but not built are in [`v2-beta/`](v2-beta/README.md).

### Contributing

- [Contributing](contributing/CONTRIBUTING.md)
- [Development setup](contributing/development-setup.md)
- [Testing](contributing/testing.md)

## Conventions in these pages

- Numbers come from `bench/results/` in this repository, and each page names the directory it
  quotes. A figure that was estimated rather than measured says so.
- The complete programs on these pages were run against the current code before publishing,
  and their sample output is real. Shorter snippets assume the setup shown earlier on the same
  page.
- "Chunk" means the unit the engine indexes and returns. "Document" means what you add: one
  `doc_id` and its text, split into chunks.
