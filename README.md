# Gyrfalcon

Gyrfalcon is a retrieval engine for RAG: BM25 and dense vector search over the same chunks,
fused, in one Rust library with a Python package and a command line. It is built to be the
fastest way to get good retrieval, including on small machines: no server, no vector database,
no GPU, and a default tier that needs no model runtime at all.

```python
from gyrfalcon import Collection

with Collection("./my-index") as col:          # created on first use; lite tier by default
    col.add_path("docs/")                       # PDF, HTML, Markdown, and text files
    col.add("note-1", "Gyrfalcons are the largest falcons.", metadata={"topic": "birds"})
    for hit in col.search("largest falcon", k=5, filter={"topic": "birds"}):
        print(f"{hit.score:.3f}  {hit.doc_id}  {hit.text[:80]}")
```

```sh
gyr index ./my-index docs/        # files and directories
gyr query ./my-index "largest falcon"
gyr serve ./my-index              # HTTP API
```

## How it compares

Each framework builds its standard hybrid retriever (BM25 plus dense, fused) over the same BEIR
datasets with the same embedder, potion-retrieval-32M, on a 4-vCPU cloud VM
(`bench/compare`, results in `bench/results/2026-10-25-cloud-4core-h2h`).

| | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| nDCG@10, SciFact | **0.709** | 0.695 | 0.604 | 0.674 |
| nDCG@10, NFCorpus | **0.344** | 0.308 | 0.254 | 0.327 |
| nDCG@10, ArguAna | **0.491** | 0.473 | 0.387 | 0.441 |
| nDCG@10, FiQA | **0.266** | 0.242 | 0.188 | 0.239 |
| index FiQA (57k docs), s | **2.00** | 32.8 | 20.6 | 17.7 |
| query p50, FiQA, ms | **2.68** | 1,436 | 2,398 | 2,226 |
| peak RSS, FiQA, MB | **414** | 2,116 | 2,791 | 2,386 |

With the full tier against the same frameworks on bge-small-en-v1.5
(`bench/results/2026-10-26-cloud-4core-bge-h2h`), SciFact scores 0.743 nDCG@10 against
0.639–0.727, indexes in 193 s against 636–1,251 s, answers in 9 ms against 137–167 ms, and
peaks at 499 MB against 1.45–1.52 GB.

Gyrfalcon's fusion weights were chosen on validation splits of SciFact, NFCorpus, ArguAna,
FiQA, and Open RAG Bench; the other frameworks run their defaults. On four BEIR sets never used
for a choice the lite tier scores TREC-COVID 0.680, SciDocs 0.165, Quora 0.834, and Touché-2020
0.318 (`bench/results/2026-10-26-quality-sweep`).

On one core the gap holds: SciFact indexes in 0.9 s against 7–19 s, and queries take 2 ms
against 110–180 ms (`bench/results/2026-10-25-cloud-1core-h2h`; Gyrfalcon's rows refreshed in
`bench/results/2026-10-26-lite-refresh`).

## Tiers

| tier | embedder | needs | for |
|---|---|---|---|
| `lite` (default) | static potion-retrieval-32M | nothing | any CPU; the numbers above |
| `full` | int8 bge-small-en-v1.5 | ONNX Runtime (`pip install gyrfalcon[full]`) | higher quality when ingest time allows |

Models download on first use into `~/.cache/gyrfalcon/models` (or `$GYR_MODELS`); the CLI
uses `scripts/fetch-models.sh`. PDFs need PDFium (`pip install gyrfalcon[pdf]`, or
`GYR_PDFIUM` pointing at the library).

## How it works

- **Lexical:** an owned BM25 index per segment (bit-packed postings with block skip data,
  one-byte length norms, scores identical to Tantivy's), scored term at a time.
  The analyzer drops English stopwords and Snowball-stems.
- **Dense:** a graph-free index. Rows are centred, rotated by a randomized Hadamard transform,
  and kept as sign bits; a query is answered by a Hamming prefilter, a 4-bit estimate, and an
  exact int8 rescore. Recall@50 is 0.996–1.000 of exact search, with no graph to build.
- **Fusion:** the lexical candidates are rescored with the dense similarity, with a weight
  chosen per tier on validation splits.
- **Filters:** document metadata is indexed as reserved terms, so a filter is a postings
  lookup; both legs then skip everything outside it, and the dense cascade scans only the
  allowed rows when they are few.
- **Writes:** small writes go to a write-ahead log and an in-memory buffer that queries see at
  once; larger ones seal immutable segments, merged in the background. Updates and deletes are
  atomic, and a crash loses at most the write in flight.
- **Files:** PDF text through PDFium plus a layout layer (running headers, page numbers,
  hyphenation, paragraphs), HTML by its block structure; chunks of 2 KB cut at paragraph and
  sentence boundaries.

The design and every change since are in `docs/superpowers/specs/2026-10-24-gyrfalcon-design.md`;
benchmarks and how to reproduce them are under `bench/`.

## Building

```sh
cargo build --release                         # the gyr binary in target/release
scripts/fetch-models.sh                       # models for the CLI and benches
cd crates/gyrfalcon-py && maturin develop --release && pytest tests   # the Python package
```

Licensed under MIT or Apache-2.0.
