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
(`2026-10-26-quality-sweep`). The same-rule comparison against the other frameworks, including
these four sets, is the next section.

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

## Same-rule comparison on all eight sets

nDCG@10, lite tier, potion-retrieval-32M, depth 100, 4 vCPU
(`bench/results/2026-10-29-heldout-frameworks`). Gyrfalcon configured is rescore-max, dense
weight 0.45. Gyrfalcon RRF and the three frameworks use reciprocal rank fusion. No weight was
chosen on this run. The tuning-set framework cells are the 2026-10-25 run and were not repeated.
The Gyrfalcon cells are the full test splits, except the Quora sample row.

| Dataset | Queries | Gyrfalcon configured | Gyrfalcon RRF | LlamaIndex | LangChain | Haystack |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.6978 | 0.6945 | 0.6040 | 0.6739 |
| NFCorpus | 323 | 0.3435 | 0.3395 | 0.3081 | 0.2543 | 0.3265 |
| ArguAna | 1,406 | 0.4912 | 0.4893 | 0.4725 | 0.3867 | 0.4414 |
| FiQA | 648 | 0.2659 | 0.2432 | 0.2420 | 0.1883 | 0.2388 |
| TREC-COVID | 50 | 0.6797 | 0.6312 | 0.6355 | 0.5433 | 0.6016 |
| SciDocs | 1,000 | 0.1646 | 0.1585 | 0.1571 | 0.1270 | 0.1563 |
| Quora | 10,000 | 0.8344 | 0.8178 | | | |
| Quora sample | 200 of 10,000 | 0.8506 | 0.8292 | 0.7900 | 0.7244 | 0.7961 |
| Touché-2020 | 49 | 0.3183 | 0.3089 | 0.2816 | 0.2815 | 0.3337 |

The frameworks were not run on all 10,000 Quora queries. The sample was drawn before any of
these scores, seed `20261029`, and stored in `quora-sample.json` (first id `100298`, last
`98114`). Its configured nDCG@10 is 0.8506, above the full-split 0.8344. The sample does not
replace the full-split number.

RRF is below the configured rule on every set. It is also below LlamaIndex on TREC-COVID
(0.6312 vs 0.6355); configured still leads that set. Haystack leads both Gyrfalcon rules on
Touché-2020 (0.3337 vs 0.3183 and 0.3089), and its recall@100 there is 0.5766 against 0.5534
and 0.5597.

This machine was Python 3.12.3. The reused 2026-10-25 cells were Python 3.13. Touché-2020 and
the Quora sample peaked at 13 to 15 GB RSS on a 15 GB machine and used a 16 GB swap file.
LangChain's Touché query p50 was 29.5 s under that swap. Those clocks are not comparable to a
run that stayed in RAM, and they are not comparable to Gyrfalcon's full-split Quora p50.
nDCG does not depend on the swap. TREC-COVID has one empty document; LlamaIndex skipped it.
The cells are library defaults. A tuned setup per framework was not added.

## Cross-encoder depth

MiniLM (`ms-marco-MiniLM-L-6-v2`) on CPU, 4 vCPU, lite tier, rescore-max dense weight 0.45.
K in {0, 8, 16, 20, 32, 50}. Official BEIR test, both hash halves. Quora is the same 200-query
sample, not the full split. Latency is replayed step time: `K` times the per-pair time measured
on the trace's union batch (about 60 to 80 pairs), not a fresh batch of K pairs. Recall@100
does not change with K. Full tier was not run. Open RAG Bench was not rerun.
`bench/results/2026-10-29-rerank-sweep`.

nDCG@10:

| Set | n | K=0 | K=8 | K=16 | K=20 | K=32 | K=50 |
|---|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.6977 | 0.6869 | 0.6816 | 0.6833 | 0.6784 |
| NFCorpus | 323 | 0.3435 | 0.3546 | 0.3533 | 0.3536 | 0.3520 | 0.3510 |
| ArguAna | 1,406 | 0.4912 | 0.5023 | 0.5002 | 0.4979 | 0.4905 | 0.4810 |
| FiQA | 648 | 0.2659 | 0.2979 | 0.3177 | 0.3189 | 0.3309 | 0.3385 |
| SciDocs | 1,000 | 0.1646 | 0.1699 | 0.1695 | 0.1663 | 0.1665 | 0.1660 |
| TREC-COVID | 50 | 0.6797 | 0.6885 | 0.7187 | 0.7232 | 0.7210 | 0.7520 |
| Quora sample | 200 | 0.8506 | 0.8513 | 0.8493 | 0.8460 | 0.8456 | 0.8444 |
| Touché-2020 | 49 | 0.3183 | 0.3227 | 0.3031 | 0.3096 | 0.2910 | 0.2852 |

Deeper K is not uniformly better. FiQA keeps rising (+0.032 at K=8, +0.073 at K=50) and
TREC-COVID is largest at K=50 (+0.072, 50 queries). NFCorpus, ArguAna, and SciDocs peak at K=8.
SciFact is worse at every K (−0.011 at K=8, −0.030 at K=50). Touché-2020 gains 0.004 at K=8 and
is 0.033 under no rerank at K=50. ArguAna at K=50 is 0.010 under no rerank. The Quora sample is
flat at K=8 and 0.006 under no rerank at K=50.

On this host a long query is 11–13 ms/pair, so K=8 is about 90 ms p50 and K=50 is about 530–650 ms.
Quora questions are 1.6 ms/pair, so K=8 is 18 ms p50. The older "~180 ms and +0.08 nDCG on FiQA"
note was about 22 ms/pair and a validation comparison, not this test-set K=8 point. Replay p50
with no rerank matched live within 10% only on SciFact and NFCorpus; on the other six it ran
high. nDCG matched live on the half the built-in checker scores.

An offline selective replay, not a planner, reranks when the configured fusion's rank-1 score
minus rank-2 is below 0.0379. That threshold was fit only on the tuning validation splits
(2,323 queries), inside a predeclared 0.005 nDCG band of always-reranking, preferring the
smallest rerank time. It reranks 549 of those queries and, at K=8, keeps 24% of the rerank
milliseconds. p50 falls because most queries are skipped; p99 stays on the rerank path. On the
test data the same frozen threshold gives up ArguAna's K=8 gain (11 of 716 hash-half queries
reranked) and about a quarter of FiQA's K=8 gain. SciFact selective (0.7173) beats both
always-rerank (0.6977) and no rerank (0.7088). At K=50 the same threshold gives up 0.026 of
FiQA's gain and 0.034 of TREC-COVID's versus always-rerank, and Touché stays well under no
rerank. Held-out queries did not move the threshold.

## Run-to-run variance

Five consecutive runs of the lite tier at the current commit, nothing else on the machine
(`bench/results/2026-10-26-lite-variance`). Each run builds every index from scratch.

| Dataset | Queries | nDCG@10, all runs | Index s, median (range) | Query p50 ms, median (range) | Query p99 ms, median (range) |
|---|---|---|---|---|---|
| SciFact | 300 | 0.7088 | 0.34 (0.30 to 0.42) | 1.62 (1.53 to 1.75) | 2.50 (2.01 to 2.84) |
| NFCorpus | 323 | 0.3435 | 0.28 (0.26 to 0.35) | 1.59 (1.48 to 1.98) | 3.26 (2.30 to 3.84) |
| ArguAna | 1,406 | 0.4912 | 0.43 (0.41 to 0.53) | 2.00 (1.87 to 2.12) | 3.90 (3.31 to 5.00) |
| FiQA | 648 | 0.2659 | 1.74 (1.61 to 1.98) | 2.54 (2.40 to 2.62) | 4.02 (3.66 to 6.35) |
| TREC-COVID | 50 | 0.6797 | 12.30 (12.06 to 12.51) | 3.83 (3.17 to 4.43) | 6.77 (5.25 to 11.57) |
| SciDocs | 1,000 | 0.1646 | 1.16 (1.11 to 1.19) | 2.20 (2.10 to 2.25) | 4.02 (3.56 to 4.49) |
| Quora | 10,000 | 0.8344 | 6.50 (6.12 to 6.98) | 5.39 (5.37 to 5.47) | 8.97 (8.68 to 9.44) |
| Touché-2020 | 49 | 0.3183 | 45.93 (45.00 to 48.11) | 5.58 (4.96 to 7.23) | 11.61 (9.24 to 20.03) |

- **Quality is deterministic.** nDCG@10 was identical to every digit in all five runs. The full
  tier is not bit-for-bit deterministic: its int8 model quantizes activations per batch, and
  SciFact moved between 0.742 and 0.748 across batching configurations.
- **Latency is stable.** Across runs, p50 varied by at most 1.15x on five of the eight sets and
  by at most 1.5x on the others. p99 stays under 12 ms on every set at the median.
- **A p99 bug was found and fixed this way.** The first variance runs showed p99 of 84 ms on
  TREC-COVID and 237 ms on Touché-2020. The cause: after a merge, the first query to finish
  deleted the merged-away segment files on its own thread, while holding a lock other queries
  wait on. Deletes now run on a background thread outside that lock (commit de49193), and the
  same sets measure 6.8 ms and 11.6 ms. The table above is from after the fix.
- **The other frameworks' cells were run once.** The gaps to them are 12 to 900 times in latency,
  well outside the variation measured here, but their own run-to-run variance is not measured.

## Reading from disk

Every latency above is measured with the collection in memory. Gyrfalcon memory-maps its files,
so after a restart, or when the collection does not fit in RAM, queries read from disk until the
operating system has cached what they touch. `bench/cold_cache.py` measures that: it drops the
collection's and the model's pages from the page cache (checked with `mincore`), then opens the
collection and queries (`bench/results/2026-10-28-cold-cache`).

Cloud block storage and a hard disk are emulated with `bench/slowfs.py`, a FUSE filesystem that adds
a delay and a bandwidth cost to every read that misses the cache. The delays were set so that the
measured random 4 KB read was 1.2 ms (typical of network block storage such as AWS gp3) and 8.4 ms
(a 7,200 rpm disk). These are emulations: a real device also has queueing and IOPS limits.

FiQA, lite tier, 79 MB collection plus the 131 MB model:

| | In memory | Cloud disk, 1.2 ms per read | Hard disk, 8.4 ms per read |
|---|---|---|---|
| Open, then first query | | 0.28 s + 0.71 s | 1.0 s + 3.1 s |
| First 10 queries, p50 | 1.6 ms | 30 ms | 119 ms |
| All 648 queries from a cold start, p50 / p99 | 1.6 / 5.0 ms | 2.8 / 32 ms | 2.5 / 128 ms |
| Every query from an empty cache, p50 | | 735 ms | 2,990 ms |

Quora, lite tier, 370 MB collection:

| | In memory | Cloud disk, 1.2 ms per read | Hard disk, 8.4 ms per read |
|---|---|---|---|
| Open, then first query | | 1.0 s + 1.9 s | 3.4 s + 7.1 s |
| First 10 queries, p50 | 6.2 ms | 414 ms | 1,753 ms |
| All 1,000 queries from a cold start, p50 / p99 | 6.2 / 11.8 ms | 12.7 / 193 ms | 11.9 / 522 ms |
| Every query from an empty cache, p50 | | 1,797 ms | 7,131 ms |

- **A running service is barely affected while the collection and the model fit.** The first
  queries after a start pay for the disk; within about ten queries the cache holds what queries
  touch, and the median over the run is within a few milliseconds of the in-memory figure, even
  on a hard disk. That stops being true when the cgroup cannot keep the pages, below.
- **A collection larger than RAM, or a process per query, pays on every query.** FiQA then takes
  0.7 s per query on cloud storage and 3 s on a hard disk.
- **The cold path reads more than it needs.** A cold FiQA query reads about 36 MB: each random row
  read pulls in its neighborhood (kernel read-around), and the reads are issued one after another.
  Issuing them together and reading only the needed pages would cut both; this is not built yet.
- **This VM's own disk** is network-backed and cached by its host, so its cold numbers depend on
  what the host has cached: a cold FiQA query took 39 to 51 ms when the host had the blocks, and
  the first open after a day took 4.4 s. The emulated profiles are repeatable; this disk is not.
- **The FUSE layer** costs about 0.3 to 0.4 ms per read on its own (measured with zero added delay),
  which is why faster SSDs are not emulated.

### Index larger than the cgroup

Quora lite (522,931 chunks, 368 MB) plus the 131 MB model, 20 judged queries, `memory.max` 128 MB
or 256 MB, `memory.swap.max=0`. `eager` is warm-open. No run was OOM-killed.
`bench/results/2026-10-29-ram-cap`.

| | open | query p50 | query p99 | hit memory.max |
|---|---:|---:|---:|---:|
| virtio, 256 MB, none | 44 ms | 12.8 ms | 42.5 ms | 0 |
| virtio, 256 MB, eager | 504 ms | 11.1 ms | 39.4 ms | 2,296 |
| virtio, 128 MB, none | 42 ms | 13.1 ms | 40.2 ms | 392 |
| virtio, 128 MB, eager | 488 ms | 13.1 ms | 45.1 ms | 3,333 |
| gp3, 256 MB, none | 382 ms | 78.3 ms | 446 ms | 0 |
| gp3, 256 MB, eager | 6.1 s | 73.6 ms | 388 ms | 2,299 |
| gp3, 128 MB, none | 377 ms | 80.7 ms | 438 ms | 394 |
| gp3, 128 MB, eager | 6.4 s | 80.0 ms | 472 ms | 3,310 |

The 128 MB cap cannot hold the model and the index. Every 128 MB run sat on the cap. 256 MB
holds the model and these 20 queries' rows when warmup is off. Eager reads the collection at
open and the cgroup reclaims it during that read, so the queries still fault. On gp3 the 6 s
open does not make p50 faster than no warmup (74 ms vs 78 ms at 256 MB, 80 ms vs 81 ms at 128 MB).
Query p50 does not cliff between the two caps on this 20-query slice. The virtio latency hides
the refaults: this disk's host cache serves them, and each query still reads 3–5 MB p50 there
(about 50 MB on the first). `read_bytes` on the FUSE mount does not include the daemon's backing
reads. No corpus of 1M+ chunks was built.

The head-to-head frameworks keep their indexes in memory and rebuild them on every start in the
benchmark setup (17.7 to 32.8 s for FiQA). Their time to load a persisted index from disk was not
measured.

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
