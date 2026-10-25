# Head-to-head, lite embedder (potion-retrieval-32M), 4 cores

Gyrfalcon rows: analyzer gyr-text-v2, default fusion rescore with dense weight 0.4 (commit after 33134c1). The other frameworks' rows are from commit 4c2b666 and do not depend on Gyrfalcon. `results-v1-analyzer.jsonl` holds the earlier Gyrfalcon rows (gyr-text-v1, dense weight 0.2), which led only on speed and memory.

The dense weight 0.4 was chosen on validation halves (`replay-lite-v2-validation.json`: best mean nDCG@10 over ORB and four BEIR sets) and confirmed on the test halves (`replay-lite-v2-test.json`: significantly better than 0.2 on all four BEIR sets, ORB -0.006 not significant).

### nDCG@10 (higher is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **0.4729** | 0.4725 | 0.3867 | 0.4414 |
| potion | 4 | fiqa | **0.2689** | 0.2420 | 0.1883 | 0.2388 |
| potion | 4 | nfcorpus | **0.3410** | 0.3081 | 0.2543 | 0.3265 |
| potion | 4 | scifact | **0.7092** | 0.6945 | 0.6040 | 0.6739 |

### Recall@100

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **0.9687** | 0.9424 | 0.9154 | 0.9566 |
| potion | 4 | fiqa | **0.5573** | 0.5544 | 0.4786 | 0.5434 |
| potion | 4 | nfcorpus | 0.3080 | 0.2982 | 0.2762 | **0.3081** |
| potion | 4 | scifact | 0.9570 | 0.9587 | 0.9392 | **0.9603** |

### Index build, seconds (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **0.58** | 5.78 | 3.47 | 3.75 |
| potion | 4 | fiqa | **2.65** | 32.80 | 20.64 | 17.65 |
| potion | 4 | nfcorpus | **0.31** | 4.06 | 2.87 | 2.27 |
| potion | 4 | scifact | **0.46** | 5.21 | 2.92 | 2.76 |

### Query p50, ms (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **2.45** | 189.69 | 666.28 | 533.71 |
| potion | 4 | fiqa | **2.95** | 1436.00 | 2397.78 | 2226.09 |
| potion | 4 | nfcorpus | **1.76** | 79.47 | 69.15 | 116.96 |
| potion | 4 | scifact | **1.91** | 107.81 | 116.17 | 165.82 |

### Query p99, ms

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **4.21** | 379.97 | 1751.89 | 996.42 |
| potion | 4 | fiqa | **4.82** | 2700.15 | 3220.52 | 3082.61 |
| potion | 4 | nfcorpus | **2.57** | 172.15 | 109.92 | 205.61 |
| potion | 4 | scifact | **3.40** | 170.59 | 236.51 | 270.08 |

### Peak RSS, MB (lower is better)

| embedder | cores | dataset | gyrfalcon | llamaindex | langchain | haystack |
|---|---|---|---|---|---|---|
| potion | 4 | arguana | **344** | 644 | 658 | 592 |
| potion | 4 | fiqa | **520** | 2116 | 2791 | 2386 |
| potion | 4 | nfcorpus | **294** | 466 | 457 | 414 |
| potion | 4 | scifact | **303** | 517 | 532 | 470 |
