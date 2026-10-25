# Head-to-head, lite embedder (potion-retrieval-32M), 1 core

Commit 04fb5b1 (gyr-text-v2, dense weight 0.4), `bench/compare/compare.py --cores 1 --datasets scifact,nfcorpus`. Every process pinned to core 0 with thread pools capped to 1.

### nDCG@10 (higher is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | **0.3410** | 0.3081 | 0.2543 | 0.3265 |
| potion | 1 | scifact | **0.7092** | 0.6945 | 0.6040 | 0.6739 |

### Recall@100

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | 0.3080 | 0.2982 | 0.2762 | **0.3081** |
| potion | 1 | scifact | 0.9570 | 0.9587 | 0.9392 | **0.9603** |

### Index build, seconds (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | **0.81** | 6.45 | 4.74 | 4.12 |
| potion | 1 | scifact | **1.14** | 18.64 | 7.09 | 7.11 |

### Query p50, ms (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | **1.79** | 78.64 | 67.54 | 117.44 |
| potion | 1 | scifact | **1.98** | 109.64 | 137.97 | 177.10 |

### Query p99, ms

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | **2.60** | 137.34 | 104.86 | 202.77 |
| potion | 1 | scifact | **3.12** | 179.50 | 237.07 | 356.36 |

### Peak RSS, MB (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 1 | nfcorpus | **294** | 452 | 442 | 401 |
| potion | 1 | scifact | **295** | 504 | 517 | 456 |
