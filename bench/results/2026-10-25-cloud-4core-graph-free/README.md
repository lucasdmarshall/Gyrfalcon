# Graph-free dense index (format version 3), 4-core cloud VM

`beir-lite-usearch.json` is commit 6318674 (owned BM25, USearch HNSW). `beir-lite-graph-free.json`
is the graph-free cascade. Both ran

    gyr bench beir --tier lite --dataset scifact nfcorpus arguana fiqa \
      --fusions dense,lexical,rescore:0.2,rescore:0.6

| dataset | fusion | nDCG@10 HNSW | graph-free | recall@100 HNSW | graph-free | index s HNSW | graph-free |
|---|---|---|---|---|---|---|---|
| scifact | dense | 0.6300 | 0.6323 | 0.9309 | 0.9342 | 0.86 | 0.35 |
| scifact | rescore:0.6 | 0.6972 | 0.6972 | 0.9603 | 0.9603 | | |
| nfcorpus | dense | 0.3018 | 0.3076 | 0.2817 | 0.2852 | 0.63 | 0.24 |
| nfcorpus | rescore:0.6 | 0.3355 | 0.3355 | 0.3047 | 0.3035 | | |
| arguana | dense | 0.4516 | 0.4516 | 0.9573 | 0.9580 | 1.52 | 0.50 |
| arguana | rescore:0.6 | 0.4677 | 0.4677 | 0.9623 | 0.9630 | | |
| fiqa | dense | 0.1831 | 0.1887 | 0.4745 | 0.4840 | 5.39 | 2.46 |
| fiqa | rescore:0.6 | 0.2567 | 0.2573 | 0.5442 | 0.5463 | | |

Recall against an exact int8 scan (`examples/profile_dense`, 1,000 queries, limit 100 unless
noted):

| corpus, model | HNSW R@50 | graph-free R@50 | HNSW build | graph-free build |
|---|---|---|---|---|
| FiQA 57,638, potion-retrieval-32M | 0.9411 | 0.9986 (limit 50: 0.9958) | 4.6 s | part of a 1.3 s ingest |
| SciFact 5,183, potion-retrieval-32M | 0.9763 | 1.0000 | 0.17 s | |
| SciFact 5,183, bge-small-en-v1.5 q8 | 0.985 | 1.0000 (prototype) | 0.18 s | |
| NFCorpus 3,633, bge-small-en-v1.5 q8 | 0.963 | 0.9997 (prototype) | 0.10 s | |

The bge rows come from the prototype cascade in the same example before it moved into the
engine. Without centering on the segment mean, bge recall@50 was 0.972 and 0.945, below HNSW;
centering is what makes one bit per dimension work for anisotropic embeddings.

FiQA dense latency at limit 50: one segment 0.56 ms, two 0.66 ms, six 0.75 ms, twenty 1.34 ms,
recall@50 0.996 to 0.999. HNSW took 0.18-0.20 ms at recall@50 0.94.
