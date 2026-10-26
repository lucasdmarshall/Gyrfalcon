<sub>[Docs](../README.md) / Guides</sub>

# Choosing a tier

A tier fixes the embedding model and the fusion rule of a collection. There are two:

| | `lite` (default) | `full` |
|---|---|---|
| Embedder | potion-retrieval-32M, a static table | bge-small-en-v1.5, int8 ONNX transformer |
| Vector width | 512 | 384 |
| Needs | nothing | ONNX Runtime |
| Fusion | BM25 over its query maximum, plus raw cosine, dense weight 0.45 | min-max BM25 and cosine, dense weight 0.65 |
| Embedding cost | a table lookup per token | a transformer pass per text |

The choice is made when a collection is created and cannot change afterwards: the collection
records its model, and opening it with the other tier is refused. Switching means rebuilding
(see the end of this page).

## The trade in numbers

Same 4-vCPU cloud VM, BEIR test splits, nDCG@10 (`bench/results/2026-10-26-quality-sweep`,
`2026-10-26-cloud-4core-bge-h2h`, `2026-10-26-lite-refresh`):

| | SciFact lite | SciFact full | NFCorpus lite | NFCorpus full | ArguAna lite | ArguAna full |
|---|---|---|---|---|---|---|
| nDCG@10 | 0.709 | **0.743** | 0.344 | **0.361** | 0.491 | **0.612** |
| Index time | **0.32 s** | 193 s | **0.28 s** | 156 s | | |
| Query p50 | **1.7 ms** | 9.1 ms | **1.6 ms** | 5.9 ms | | |
| Peak RSS | **268 MB** | 499 MB | **258 MB** | 486 MB | | |

SciFact has 5,183 documents, NFCorpus 3,633. On 4 cores the full tier indexes about 25
documents per second; the lite tier indexes 13,000 to 29,000.

Two things stand out:

1. **Quality.** The full tier is 0.02 to 0.12 nDCG@10 better. The gap is widest where queries
   and relevant documents share few words (ArguAna, where a query is an argument and the answer
   is its counter-argument), and narrowest where wording overlaps (NFCorpus, SciFact).
2. **Ingest.** The full tier is 500 to 600 times slower to index. Query latency differs far less,
   because a query is embedded once and short texts are cheap for the transformer.

## Use `lite` when

- You index often, or index a lot: logs, tickets, chat history, crawls, anything that changes
  hourly or reaches hundreds of thousands of documents.
- The machine is small. The lite tier indexes SciFact in 0.9 s on one core
  (`bench/results/2026-10-26-lite-refresh`).
- You cannot ship ONNX Runtime, or you want the smallest dependency set.
- Queries share vocabulary with the documents: technical documentation, code, product
  catalogs, support articles. BM25 carries much of the ranking there, and the lite tier's BM25
  is the same as the full tier's.

## Use `full` when

- The corpus is modest or changes rarely, so a slow index build is paid once. At about 25
  documents per second on 4 cores, 100,000 short documents take about an hour.
- Queries are paraphrases of the documents rather than keyword lookups: questions in natural
  language over prose, arguments, research abstracts.
- Ten milliseconds per query is acceptable.

If you are unsure, start with `lite`. It is the default because it gives most of the quality
for a small fraction of the cost, and a later move to `full` is a rebuild, not a code change.

## Measure on your own data

The benchmark sets above are not your corpus. Build both and compare on a few dozen real
queries:

```python
from gyrfalcon import Collection

queries = ["...", "..."]                  # questions your users actually ask
for tier in ("lite", "full"):
    with Collection(f"./eval-{tier}", tier=tier) as col:
        col.add_path("docs/")
        for q in queries:
            print(tier, q, [h.doc_id for h in col.search(q, k=5)])
```

## Reranking

The command line and the Rust API can add a cross-encoder (ms-marco-MiniLM-L-6-v2) after
fusion: `gyr query --rerank --reranker-dir DIR`, or `Index::set_reranker`. It is off by default
and not exposed in the Python package. On a 4-core CPU it cost about 22 ms per scored pair in
the [engineering review](../superpowers/analysis/2026-10-24-engineering-review.md), so scoring
the top 8 adds about 180 ms. It helped FiQA by 0.08 nDCG@10 and hurt Open RAG Bench by 0.06 to
0.09, so measure before enabling it.

## Switching tiers

A collection cannot change tier in place. Build a new one from the same sources:

```python
from gyrfalcon import Collection

with Collection("./my-index-full", tier="full") as col:
    col.add_path("docs/")
```

Opening the old collection with the other tier raises an error naming the field that differs:

```text
RuntimeError: incompatible index: model_id is minishlab/potion-retrieval-32M in the manifest, Xenova/bge-small-en-v1.5-q8 requested
```

---

<sub>Related: [Tiers](../concepts/tiers.md) · [Design record 0002](../design/0002-tier-system.md) · Next: [Indexing large corpora](indexing-large-corpora.md)</sub>
