<sub>[Docs](../README.md) / Benchmarks</sub>

# Methodology

How the numbers in [Results](results.md) were produced, what each one includes, and everything a
careful reader should know before trusting them. If a claim in these docs cannot be traced to a
procedure on this page and a file under `bench/results/`, treat it as unsupported.

## Principles

1. **Same inputs for every system.** Every framework gets the same documents, the same queries,
   the same embedding model, the same retrieval depth, and the same evaluator.
2. **Choose on one set of queries, report on another.** Gyrfalcon's tunable settings were chosen
   on validation queries. Results are reported on test queries and on datasets that were never
   used for any choice.
3. **Report uncertainty.** Quality differences carry paired bootstrap intervals. Latency carries
   repeated runs.
4. **Publish the harness.** Everything runs from this repository with the commands in
   [Reproducing](reproducing.md).

## Datasets

All from [BEIR](https://github.com/beir-cellar/beir), English, test judgments unless noted.

| Dataset | Documents | Test queries | Domain | Role |
|---|---|---|---|---|
| SciFact | 5,183 | 300 | scientific claims | tuning |
| NFCorpus | 3,633 | 323 | nutrition and medicine | tuning |
| ArguAna | 8,674 | 1,406 | counter-arguments | tuning |
| FiQA-2018 | 57,638 | 648 | financial questions | tuning |
| TREC-COVID | 171,332 | 50 | COVID-19 literature | held out |
| SciDocs | 25,657 | 1,000 | citation prediction | held out |
| Quora | 522,931 | 10,000 | duplicate questions | held out |
| Touché-2020 | 382,545 | 49 | argument retrieval | held out |

Open RAG Bench (vectara/open_ragbench, the 1,914 text-only questions over arXiv papers) is a
fifth tuning set, used for choices and engineering checks but not in the head-to-head tables.

A document is its title and text joined by a blank line. In the BEIR runs each document is
indexed as one unit, not chunked, for every system, so the comparison measures retrieval and not
chunking policy. Both embedders read at most 512 tokens of a document.

## Splits: tuning and held-out

| Dataset | Choices made on | Reported on |
|---|---|---|
| SciFact | `train` judgments | `test` |
| NFCorpus, FiQA | `dev` judgments | `test` |
| ArguAna, Open RAG Bench | the half of `test` queries whose `sha256(query_id)` starts with an even byte | the other half (in planner reports); full `test` (in head-to-head tables) |
| TREC-COVID, SciDocs, Quora, Touché-2020 | never used | `test` |

The four held-out sets were added on 2026-10-26, before the last round of choices, precisely so
that the final defaults could be checked on data that had no say in them. Results on them are
the best available estimate of how the defaults behave on a new corpus.

## Metrics

**Quality.** nDCG@10 with linear gain and recall@100, as computed by `pytrec_eval` and reported
by BEIR and MTEB. Duplicate document ids in a ranking are dropped, and a retrieved document whose
id equals the query id is ignored (ArguAna's convention). One evaluator scores every system's
run file (`bench/compare/common.py`); it reproduces `gyr bench beir` to within 1e-15.

**Index time.** Wall time to build the collection from documents already in memory, after the
embedding model is loaded. For Gyrfalcon it includes the background merge and the final seal.

**Query latency.** Wall time of one complete search call per query: embedding the query, both
retrieval legs, fusion, and returning the hits. Queries run one at a time, every test query once,
after the index is built. p50 and p99 over all test queries. Model loading and index opening are
excluded on every system. The collection is in memory: the other frameworks keep their indexes in
the Python heap by design, and Gyrfalcon's memory-mapped files are in the page cache because they
were just written. Latency when Gyrfalcon reads from disk is measured separately
([Results](results.md#reading-from-disk)).

**Peak memory.** The process's peak resident set size (`ru_maxrss`). It includes the harness's
own copy of the corpus and, for Gyrfalcon, memory-mapped file pages that were touched.

## The head-to-head setup

Each framework builds its standard hybrid retriever from its own documented components:

| System | Lexical | Dense store | Fusion |
|---|---|---|---|
| LlamaIndex 0.14.25 | `BM25Retriever` (bm25s, English stemming) | `VectorStoreIndex`, in memory | `QueryFusionRetriever`, reciprocal rank, one query |
| LangChain (core 1.7.0, classic 1.0.8) | `BM25Retriever` (rank_bm25) | `InMemoryVectorStore` | `EnsembleRetriever`, weights 0.5 and 0.5 |
| Haystack 3.4.1 | `InMemoryBM25Retriever` | `InMemoryEmbeddingRetriever` | `DocumentJoiner`, reciprocal rank |
| Gyrfalcon | owned BM25, `gyr-text-v2` analyzer | graph-free int8 index | the tier's rule |

- **Embedders.** All three Python frameworks receive one shared embedder object: potion-
  retrieval-32M through `model2vec` 0.10.0, or bge-small-en-v1.5 through `fastembed` 0.9.1.
  Gyrfalcon runs the same models through its own code.
- **Depth.** Each leg returns 100 documents and the fused list keeps 100.
- **Isolation.** Each cell (one system, one dataset, one core count) runs in a fresh process
  pinned with `taskset` to N cores, with every thread pool (`OMP_NUM_THREADS`,
  `RAYON_NUM_THREADS`, ONNX Runtime) capped to N.
- **Machine.** A shared cloud VM with 4 vCPUs (Intel Xeon @ 2.10 GHz), Linux, Python 3.13.

## Statistics

Quality comparisons between two configurations use a paired bootstrap over queries: 10,000
resamples of the per-query difference, seed 1, 95% interval. A difference whose interval contains
zero is reported as no significant difference. In `bench/results/` tables a `*` marks a
significant difference.

Latency is reported as the median of five runs with the range across runs; see
[Results](results.md#run-to-run-variance).

## Disclosures

These points affect how the comparison should be read. They are listed here rather than buried.

1. **Gyrfalcon's fusion weights were tuned; the other frameworks run their defaults.** The
   weights were chosen on validation queries of SciFact, NFCorpus, ArguAna, FiQA, and Open RAG
   Bench, which are different queries from the test queries reported, over the same documents.
   The held-out sets carry no such advantage, which is why they are reported separately.
2. **Part of the quality lead is the fusion rule, not the engine.** With the full tier and plain
   reciprocal rank fusion on SciFact, Gyrfalcon scores 0.7266 and LlamaIndex 0.7270: the same.
   Gyrfalcon's default rule raises that to 0.743. On NFCorpus Gyrfalcon leads even under the same
   rule (0.363 against at most 0.357), which comes from its BM25 analyzer.
3. **LangChain's BM25 tokenization.** LangChain's `BM25Retriever` splits on whitespace by default,
   without lowercasing or stemming. This probably explains much of its lower quality. Per-leg
   scores were not measured for the other frameworks, so this is an explanation, not a
   measurement.
4. **Quantized bge builds differ.** For the full-tier comparison the Python frameworks use
   fastembed's int8 export of bge-small-en-v1.5 and Gyrfalcon uses Xenova's. Same model, different
   quantized files.
5. **Other frameworks ran once per cell.** Gyrfalcon's lite cells have five repeated runs; the
   other frameworks' cells have one. The LlamaIndex SciFact bge cell ran while other builds shared
   the machine, so its index time (1,251 s) may be high. The latency gaps, 12 to 900 times, are
   far larger than run-to-run noise, but individual cells carry that caveat.
6. **Rows come from different commits.** The other frameworks' rows do not depend on Gyrfalcon's
   code. Gyrfalcon's rows were refreshed after later engine changes; each results directory names
   its commit.
7. **The machine is shared.** Cloud VMs have neighbors, so a single run can be slow for reasons
   outside the program; repeated runs are the defense. On TREC-COVID and Touché-2020, with about
   50 queries each, p99 is in effect the slowest query and moves more between runs than p50 does.
   Reports from before commit de49193 carry a p99 bug on those two sets (a query deleting merged
   files inline), fixed since ([Results](results.md#run-to-run-variance)).
8. **Head-to-head latency is from memory.** Every system answered from RAM. Gyrfalcon from disk is
   measured on its own; the other frameworks' time to load a persisted index from disk is not
   measured.
9. **What is not measured.** End-to-end answer quality with an LLM, non-English text, a corpus of
   1M+ chunks (Quora lite is 522,931), GPUs, the full tier's rerank sweep, Open RAG Bench under
   this reranker, the three frameworks on all 10,000 Quora queries, a tuned setup beside each
   framework's defaults, and per-leg quality of the other frameworks.
10. **The eight-set framework table mixes two Python versions and one reused run.** Tuning-set
    framework cells are the 2026-10-25 measurement (Python 3.13) and were not repeated. Held-out
    framework cells are Python 3.12.3 on 2026-10-29. Quora's framework cells are a 200-query
    sample drawn with seed `20261029` before scoring and stored in `quora-sample.json`; that
    sample's Gyrfalcon nDCG is higher than the full split and is not the official Quora number.
    Touché-2020 and the Quora sample used a 16 GB swap file. Their nDCG stands; their latency
    does not. TREC-COVID has one empty document, which LlamaIndex skipped.
11. **Rerank latency in the depth sweep is replayed.** The per-pair time was measured on a union
    batch of about 60 to 80 pairs, and a depth of K is costed as K times that time. Replay p50
    with no rerank was within 10% of live only on SciFact and NFCorpus. The selective threshold
    was fit on tuning validation only, with a predeclared 0.005 nDCG band, and was not refit on
    test or held-out queries. No planner was built from it. The full tier was not run.
12. **A cgroup cap is not the host cache.** The 128 MB and 256 MB Quora runs set `memory.swap.max`
    to 0. The virtio numbers are this VM's host-cached disk: guest pages were dropped, the host's
    copy was not, so latency stays low while bytes read do not. The gp3 numbers are the FUSE
    profile (`--latency-ms 0.65 --mb-per-s 125`), which does not model IOPS. Eager warmup under
    the cap is reclaimed during open and does not make the following queries faster.

---

<sub>Previous: [Results](results.md) · Next: [Reproducing](reproducing.md)</sub>
