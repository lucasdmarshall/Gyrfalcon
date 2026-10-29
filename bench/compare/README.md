# Head-to-head against other RAG frameworks

Each framework builds its own standard hybrid retriever (BM25 plus dense, fused) over a BEIR
dataset; Gyrfalcon runs `gyr bench beir` with its default fusion. Every framework gets the same
embedder object (`common.Embedder`), so differences come from the framework, not the model:

| system | lexical | dense store | fusion |
|---|---|---|---|
| LlamaIndex | `BM25Retriever` (bm25s, English stemming) | `VectorStoreIndex` (in memory) | `QueryFusionRetriever`, reciprocal rank |
| LangChain | `BM25Retriever` (rank_bm25) | `InMemoryVectorStore` | `EnsembleRetriever`, weighted RRF |
| Haystack | `InMemoryBM25Retriever` | `InMemoryEmbeddingRetriever` | `DocumentJoiner`, RRF |
| `langchain-stem` | the LangChain retriever, with lowercase, `\b\w\w+\b`, and an English Snowball stemmer | same as LangChain | same as LangChain |
| `haystack-stem` | Haystack's own BM25 regex, then the same stemmer | same as Haystack | same as Haystack |

`langchain-stem` and `haystack-stem` sit beside the defaults. They do not replace them.
LlamaIndex's default already stems, so it has no stem variant. Numbers:
`bench/results/2026-10-29-tuned-baselines`.

Embedders: `potion` (Model2Vec `potion-retrieval-32M`, Gyrfalcon's lite tier) and `bge-small`
(`BAAI/bge-small-en-v1.5` ONNX through fastembed; Gyrfalcon's full tier runs it through ort).

Every cell is one process pinned with `taskset` to N cores, with thread pools capped to N; peak
RSS is that process's `ru_maxrss`. All run files are scored by `common.evaluate` (pytrec_eval
nDCG@10 with linear gain, recall@100), which reproduces `gyr bench beir` to 1e-15.

A full judged set that is too slow to query can be cut to a fixed-seed sample. The sample is
drawn from the sorted judged queries with `random.Random(seed).sample`, then sorted by id.
The report records `subsample.query_ids`; those ids are the sample. `gyr bench beir` still
runs every judged query, and `compare.py` scores the sample out of that run file. Its latency
fields stay the full-split numbers (`gyr_latency_is_full_split`).

    python run_framework.py haystack BEIR_DIR potion 4 OUT_PREFIX 200 20261029
    python compare.py OUT_DIR --datasets quora --systems llamaindex --max-queries 200 --seed 20261029

    python -m venv .venv-compare
    .venv-compare/bin/pip install fastembed model2vec psutil llama-index-core \
        llama-index-retrievers-bm25 langchain-core langchain-community rank_bm25 haystack-ai
    .venv-compare/bin/python bench/compare/compare.py OUT_DIR --cores 4 --datasets scifact,nfcorpus
    .venv-compare/bin/python bench/compare/summarize.py OUT_DIR/results.jsonl

Versions measured: llama-index-core 0.14.25, langchain-core 1.7.0 (langchain-classic 1.0.8),
haystack-ai 3.4.1, fastembed 0.9.1, model2vec 0.10.0, Python 3.13.
