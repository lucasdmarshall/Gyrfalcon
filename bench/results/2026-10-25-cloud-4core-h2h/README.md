# Head-to-head, lite embedder (potion-retrieval-32M), 4 cores

Commit 4c2b666, `bench/compare/compare.py --cores 4 --embedders potion`. Gyrfalcon uses its default fusion (rescore, dense weight 0.2) and the gyr-text-v1 analyzer.

### nDCG@10 (higher is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | 0.4582 | **0.4725** | 0.3867 | 0.4414 |
| potion | 4 | fiqa | 0.2419 | **0.2420** | 0.1883 | 0.2388 |
| potion | 4 | nfcorpus | 0.3218 | 0.3081 | 0.2543 | **0.3265** |
| potion | 4 | scifact | 0.6818 | **0.6945** | 0.6040 | 0.6739 |

### Recall@100

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **0.9673** | 0.9424 | 0.9154 | 0.9566 |
| potion | 4 | fiqa | 0.5432 | **0.5544** | 0.4786 | 0.5434 |
| potion | 4 | nfcorpus | 0.3027 | 0.2982 | 0.2762 | **0.3081** |
| potion | 4 | scifact | **0.9603** | 0.9587 | 0.9392 | **0.9603** |

### Index build, seconds (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **0.43** | 5.78 | 3.47 | 3.75 |
| potion | 4 | fiqa | **2.18** | 32.80 | 20.64 | 17.65 |
| potion | 4 | nfcorpus | **0.24** | 4.06 | 2.87 | 2.27 |
| potion | 4 | scifact | **0.37** | 5.21 | 2.92 | 2.76 |

### Query p50, ms (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **2.42** | 189.69 | 666.28 | 533.71 |
| potion | 4 | fiqa | **3.33** | 1436.00 | 2397.78 | 2226.09 |
| potion | 4 | nfcorpus | **1.74** | 79.47 | 69.15 | 116.96 |
| potion | 4 | scifact | **2.63** | 107.81 | 116.17 | 165.82 |

### Query p99, ms

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **4.36** | 379.97 | 1751.89 | 996.42 |
| potion | 4 | fiqa | **6.47** | 2700.15 | 3220.52 | 3082.61 |
| potion | 4 | nfcorpus | **2.11** | 172.15 | 109.92 | 205.61 |
| potion | 4 | scifact | **3.74** | 170.59 | 236.51 | 270.08 |

### Peak RSS, MB (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **359** | 644 | 658 | 592 |
| potion | 4 | fiqa | **555** | 2116 | 2791 | 2386 |
| potion | 4 | nfcorpus | **294** | 466 | 457 | 414 |
| potion | 4 | scifact | **314** | 517 | 532 | 470 |
