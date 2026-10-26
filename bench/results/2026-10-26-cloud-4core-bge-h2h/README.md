# Head-to-head, full-quality embedder (bge-small-en-v1.5), 4 cores

`bench/compare/compare.py --cores 4 --datasets scifact,nfcorpus --embedders bge-small`. Gyrfalcon
runs its full tier (int8 bge-small-en-v1.5 through ONNX Runtime, length-sorted 8 KB batches,
rescore fusion with dense weight 0.65) at commit 816b8e4; the other frameworks embed with
`BAAI/bge-small-en-v1.5` through fastembed. Every cell is one process pinned to cores 0-3.
The LlamaIndex SciFact cell ran earlier the same day while builds shared the machine, so its index
time may be somewhat high; the rest ran in one quiet window.

Gyrfalcon leads on quality, index time, latency, and memory in every cell but NFCorpus
recall@100, where Haystack is 0.004 ahead.


### nDCG@10 (higher is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | **0.3620** | 0.3372 | 0.2762 | 0.3574 |
| bge-small | 4 | scifact | **0.7431** | 0.7270 | 0.6389 | 0.7138 |

### Recall@100

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | 0.3104 | 0.3072 | 0.2883 | **0.3145** |
| bge-small | 4 | scifact | **0.9683** | 0.9650 | 0.9450 | 0.9650 |

### Index build, seconds (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | **156.20** | 449.65 | 466.17 | 465.65 |
| bge-small | 4 | scifact | **193.43** | 1251.31 | 635.88 | 656.98 |

### Query p50, ms (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | **5.85** | 71.75 | 104.95 | 127.53 |
| bge-small | 4 | scifact | **9.11** | 158.57 | 137.25 | 166.53 |

### Query p99, ms

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | **9.71** | 118.53 | 152.22 | 191.10 |
| bge-small | 4 | scifact | **13.45** | 254.03 | 201.93 | 285.34 |

### Peak RSS, MB (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| bge-small | 4 | nfcorpus | **486** | 1469 | 1443 | 1398 |
| bge-small | 4 | scifact | **499** | 1520 | 1513 | 1451 |
