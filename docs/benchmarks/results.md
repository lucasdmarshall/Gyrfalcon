<sub>[Docs](../README.md) / Benchmarks</sub>

# Benchmark results

Quality, speed, and memory of Gyrfalcon against LlamaIndex, LangChain, and Haystack on public
BEIR datasets, with the same embedding model for every system. Read
[Methodology](methodology.md) for what each number includes and the disclosures that come with
it; [Reproducing](reproducing.md) has the commands.

Machine for every table: a shared cloud VM with 4 vCPUs (Intel Xeon @ 2.10 GHz), Linux. In the
head-to-head tables the best value in each row is in bold.

## At a glance

FiQA-2018: 57,638 financial documents, 648 test queries, 4 cores.

| | Gyrfalcon lite | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| nDCG@10 | **0.266** | 0.242 | 0.188 | 0.239 |
| Index build | **2.0 s** | 32.8 s | 20.6 s | 17.7 s |
| Query p50 | **2.7 ms** | 1,436 ms | 2,398 ms | 2,226 ms |
| Peak memory | **414 MB** | 2,116 MB | 2,791 MB | 2,386 MB |

## Head-to-head

### Lite embedder (potion-retrieval-32M), 4 cores

Every system embeds with potion-retrieval-32M. Sources: `bench/results/2026-10-25-cloud-4core-h2h`
for the other frameworks; for Gyrfalcon, `2026-10-26-quality-sweep` (nDCG@10 and recall@100 at the
current defaults, commit 90f1580) and `2026-10-26-lite-refresh` (speed and memory, through the same
harness, commit 621dc84). The current defaults left speed unchanged; the
[variance runs](#run-to-run-variance) below measure it again at the current commit.

**nDCG@10** (higher is better)

| Dataset | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact | **0.7088** | 0.6945 | 0.6040 | 0.6739 |
| NFCorpus | **0.3435** | 0.3081 | 0.2543 | 0.3265 |
| ArguAna | **0.4912** | 0.4725 | 0.3867 | 0.4414 |
| FiQA | **0.2659** | 0.2420 | 0.1883 | 0.2388 |

**Recall@100** (higher is better)

| Dataset | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact | 0.9282 | 0.9587 | 0.9392 | **0.9603** |
| NFCorpus | 0.2889 | 0.2982 | 0.2762 | **0.3081** |
| ArguAna | **0.9680** | 0.9424 | 0.9154 | 0.9566 |
| FiQA | 0.5514 | **0.5544** | 0.4786 | 0.5434 |

The lite tier's fusion rule trades a little recall deep in the list for precision at the top:
before it, Gyrfalcon's recall@100 was 0.957 on SciFact and 0.308 on NFCorpus (see
[design record 0001](../design/0001-hybrid-fusion.md)). A RAG prompt usually holds the top 5 to
10 chunks, which is what nDCG@10 measures.

**Index build, seconds** (lower is better)

| Dataset | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact | **0.32** | 5.21 | 2.92 | 2.76 |
| NFCorpus | **0.28** | 4.06 | 2.87 | 2.27 |
| ArguAna | **0.44** | 5.78 | 3.47 | 3.75 |
| FiQA | **2.00** | 32.80 | 20.64 | 17.65 |

**Query latency, p50 / p99, milliseconds** (lower is better)

| Dataset | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact | **1.70 / 2.92** | 107.81 / 170.59 | 116.17 / 236.51 | 165.82 / 270.08 |
| NFCorpus | **1.60 / 2.77** | 79.47 / 172.15 | 69.15 / 109.92 | 116.96 / 205.61 |
| ArguAna | **2.08 / 3.71** | 189.69 / 379.97 | 666.28 / 1,751.89 | 533.71 / 996.42 |
| FiQA | **2.68 / 4.43** | 1,436.00 / 2,700.15 | 2,397.78 / 3,220.52 | 2,226.09 / 3,082.61 |

**Peak memory, MB** (lower is better)

| Dataset | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact | **268** | 517 | 532 | 470 |
| NFCorpus | **258** | 466 | 457 | 414 |
| ArguAna | **282** | 644 | 658 | 592 |
| FiQA | **414** | 2,116 | 2,791 | 2,386 |

### Lite embedder, 1 core

The same comparison with every process pinned to one core and every thread pool capped at one
(`2026-10-25-cloud-1core-h2h`; Gyrfalcon's cells from `2026-10-26-lite-refresh`). nDCG@10 is the
same as on 4 cores.

| | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| SciFact index build, s | **0.92** | 18.64 | 7.09 | 7.11 |
| SciFact query p50, ms | **2.18** | 109.64 | 137.97 | 177.10 |
| SciFact peak memory, MB | **224** | 504 | 517 | 456 |
| NFCorpus index build, s | **0.73** | 6.45 | 4.74 | 4.12 |
| NFCorpus query p50, ms | **1.74** | 78.64 | 67.54 | 117.44 |
| NFCorpus peak memory, MB | **206** | 452 | 442 | 401 |

On one core Gyrfalcon also indexed ArguAna in 1.38 s and FiQA in 6.14 s, at p50 2.72 ms and
3.74 ms; the other frameworks' 1-core cells for those two sets were not run.

### Full-quality embedder (bge-small-en-v1.5), 4 cores

Gyrfalcon's full tier against the same frameworks, all embedding with bge-small-en-v1.5
(`2026-10-26-cloud-4core-bge-h2h`, Gyrfalcon at commit 816b8e4; Gyrfalcon's nDCG@10 and recall@100
at the current defaults from `2026-10-26-quality-sweep/full-after.json`).

| SciFact | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| nDCG@10 | **0.7430** | 0.7270 | 0.6389 | 0.7138 |
| Recall@100 | **0.9683** | 0.9650 | 0.9450 | 0.9650 |
| Index build, s | **193** | 1,251 | 636 | 657 |
| Query p50 / p99, ms | **9.1 / 13.5** | 158.6 / 254.0 | 137.3 / 201.9 | 166.5 / 285.3 |
| Peak memory, MB | **499** | 1,520 | 1,513 | 1,451 |

| NFCorpus | Gyrfalcon | LlamaIndex | LangChain | Haystack |
|---|---|---|---|---|
| nDCG@10 | **0.3608** | 0.3372 | 0.2762 | 0.3574 |
| Recall@100 | 0.3099 | 0.3072 | 0.2883 | **0.3145** |
| Index build, s | **156** | 450 | 466 | 466 |
| Query p50 / p99, ms | **5.9 / 9.7** | 71.8 / 118.5 | 105.0 / 152.2 | 127.5 / 191.1 |
| Peak memory, MB | **486** | 1,469 | 1,443 | 1,398 |

With reciprocal rank fusion, the rule LlamaIndex uses, Gyrfalcon's full tier scores 0.7266 on
SciFact, level with LlamaIndex's 0.7270. The SciFact lead comes from Gyrfalcon's tuned fusion
rule; see [Methodology](methodology.md#disclosures).

## Held-out datasets

Four BEIR sets never used for any choice, lite tier, test splits, nDCG@10
(`2026-10-26-quality-sweep`). Other frameworks were not run on these sets.

| Dataset | Documents | BM25 only | Dense only | Gyrfalcon fused | Previous default (rescore 0.4) |
|---|---|---|---|---|---|
| TREC-COVID | 171,332 | 0.6041 | 0.4404 | **0.6797** | 0.6441 |
| SciDocs | 25,657 | 0.1561 | 0.1375 | **0.1646** | 0.1643 |
| Quora | 522,931 | 0.8085 | 0.7838 | **0.8344** | 0.8319 |
| Touché-2020 | 382,545 | **0.3366** | 0.2109 | 0.3183 | 0.3284 |
| Mean | | | | **0.4993** | 0.4922 |

Against the previous default, the paired bootstrap found the TREC-COVID and Quora gains
significant and the Touché-2020 drop not significant. Touché-2020 is the one set of eight where
fusion scored below BM25 alone.

## Run-to-run variance

Five consecutive runs of the lite tier at the current commit, nothing else on the machine
(`bench/results/2026-10-26-lite-variance`). Each run builds every index from scratch.

| Dataset | Queries | nDCG@10, all runs | Index s, median (range) | Query p50 ms, median (range) | Query p99 ms, median (range) |
|---|---|---|---|---|---|
| SciFact | 300 | 0.7088 | 0.37 (0.32 to 0.42) | 1.77 (1.72 to 1.85) | 4.06 (2.65 to 5.40) |
| NFCorpus | 323 | 0.3435 | 0.28 (0.27 to 0.31) | 1.52 (1.47 to 1.70) | 2.63 (2.10 to 4.58) |
| ArguAna | 1,406 | 0.4912 | 0.43 (0.39 to 0.58) | 2.05 (1.96 to 2.30) | 4.38 (3.67 to 5.37) |
| FiQA | 648 | 0.2659 | 1.81 (1.70 to 2.02) | 2.66 (2.45 to 2.76) | 4.48 (3.87 to 4.90) |
| TREC-COVID | 50 | 0.6797 | 12.97 (11.47 to 14.15) | 3.50 (2.99 to 3.73) | 84.34 (82.29 to 96.74) |
| SciDocs | 1,000 | 0.1646 | 1.10 (1.06 to 1.21) | 2.26 (2.15 to 2.28) | 4.39 (4.13 to 4.87) |
| Quora | 10,000 | 0.8344 | 6.81 (6.60 to 8.00) | 5.70 (5.64 to 6.02) | 9.81 (9.50 to 10.80) |
| Touché-2020 | 49 | 0.3183 | 44.09 (41.95 to 45.30) | 5.22 (5.00 to 7.17) | 237.29 (207.74 to 409.84) |

- **Quality is deterministic.** nDCG@10 was identical to every digit in all five runs. The full
  tier is not bit-for-bit deterministic: its int8 model quantizes activations per batch, and
  SciFact moved between 0.742 and 0.748 across batching configurations.
- **Latency is stable.** p50 varied by at most 1.25x across runs on seven of the eight sets, and
  1.4x on Touché-2020.
- **The p99 outliers on TREC-COVID and Touché-2020** come from a few slow queries in sets of
  about 50, where p99 is effectively the slowest query. They appear only in the first rule measured
  on a freshly built index: re-measured on the same index, the same queries have p99 of 5 to 13 ms.
  The cause is not isolated yet. p50 is the reliable figure for those two sets.
- **The other frameworks' cells were run once.** The gaps to them are 12 to 900 times in latency,
  well outside the variation measured here, but their own run-to-run variance is not measured.

## Memory in more detail

| Measurement | Value | Source |
|---|---|---|
| One-shot `gyr query` on a lite collection | 27 MB peak, 33 ms | spec, "Memory: a mapped embeddings table" |
| Indexing 100 PDFs (4,962 chunks) | 246 MB peak | same section |
| Indexing FiQA, lite, 4 cores | 414 MB peak, including the harness's copy of the corpus | `2026-10-26-lite-refresh` |
| Indexing SciFact, full tier, 4 cores | 497 MB peak | spec, "Embedding batches sized for the model" |

The lite model's 129 MB table is memory-mapped, so a process pays only for the rows of the tokens
it meets. Segment files are mapped too, and count toward resident memory only while their pages
are in use.

---

<sub>Next: [Methodology](methodology.md)</sub>
