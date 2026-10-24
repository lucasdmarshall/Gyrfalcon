# Gyrfalcon v2 plan: fastest, high-quality RAG on low CPU

Goal: a claim we can publish and anyone can reproduce: "the fastest RAG retrieval framework,
with top-tier quality, even on a 1–2 core machine".

Every claim in the README links to a report under `bench/results/` produced by a `gyr bench`
command and a script that downloads its inputs.

## Evidence so far (4-core cloud VM, Open RAG Bench text-only, 1914 questions)

| Config | index + query | recall@5 | MRR | nDCG@10 |
|---|---|---|---|---|
| v1 default: bge-small fp32 + RRF | ~38 min | 0.902 | 0.726 | 0.783 |
| lite: potion-retrieval-32M + rescore | 36 s | 0.905 | 0.752 | 0.801 |
| full: bge-small int8 + rescore | 26 min | 0.921 | 0.773 | 0.819 |

Lite: 1,380 chunks/s embed, 0.72 ms p50 retrieve, 3,349 qps at in-flight 4.

## M7: tiers and defaults

- `gyr --tier lite` (default): static `minishlab/potion-retrieval-32M`, `Fusion::Rescore`
  with dense weight 0.2.
- `gyr --tier full`: ort `bge-small-en-v1.5` int8, `Fusion::Rescore` with dense weight 0.3.
- Explicit flags override the tier. Models resolve under `$GYR_MODELS` (default `.models`);
  `scripts/fetch-models.sh` fetches both tiers.
- Library default fusion becomes `Rescore { dense_weight: 0.2 }`. RRF stays available.
- Spec gets a v2 section recording the change and the measurements behind it.

## M8: reproducible evidence

- `gyr bench beir --dataset DIR`: BEIR corpus/queries/qrels (test split), nDCG@10 and
  recall@100 with the official graded-relevance formula, so numbers compare with MTEB/BEIR.
  Datasets: SciFact, NFCorpus, FiQA-2018, ArguAna. `scripts/fetch-beir.sh` downloads them.
- Constrained-CPU rows: each bench runs under 1, 2, and 4 cores (`taskset`), ort and
  tokenizer threads capped to match.
- RAM bench measures RSS in a fresh process that only opens the collection.

## M9: head-to-head

- Same machine, same corpus, same embedding model where the framework allows it, plus each
  framework's documented default. Report ingest throughput, retrieve p50/p99, qps, nDCG@10.
- Candidates: LlamaIndex, Haystack, LangChain + a vector store, Swiftide, RustyRAG.
- Answer-correctness table with an LLM judge once a generator and judge endpoint are given.

## M10: production and adoption

- API-key auth, multiple collections per server, Docker image, Python bindings (PyO3),
  README with the reproducible numbers.
