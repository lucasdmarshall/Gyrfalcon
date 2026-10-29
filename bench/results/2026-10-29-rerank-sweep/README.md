# MiniLM rerank depth, lite tier, 2026-10-29

Cross-encoder depth sweep for the question "how does the latency/quality trade-off look when reranking the top 20 to 50 hybrid candidates?" The previous published point was the top 8 only (about 180 ms, +0.08 nDCG@10 on FiQA validation, −0.06 to −0.09 on Open RAG Bench). This directory is the lite tier. The full tier (bge-small-en-v1.5 int8, rescore dense weight 0.65) was not run.

No adaptive planner was added. The selective numbers are an offline replay of the traces.

## Setup

- Host: 4 vCPU Intel Xeon (KVM), MiniLM cross-encoder `ms-marco-MiniLM-L-6-v2` on CPU, `RAYON_NUM_THREADS=4`. One quiet run. An earlier SciFact trace taken while another job shared a core (16.82 ms/pair) was discarded and is not in this directory. The SciFact file here is the rerun (mean 11.70 ms/pair on the 300 test queries).
- Wall clock of the quiet trace: 23:34 Asia/Rangoon on 29 Oct 2026 through 00:46 Asia/Rangoon on 30 Oct 2026.
- Fusion: `RescoreMax`, dense weight 0.45, the lite configuration. Depth K in {0, 8, 16, 20, 32, 50}. K=0 is that fusion with no rerank.
- Traces: `gyr bench trace --rerank-k 50 --depth 100 --tier lite`. Raw jsonl is under `lite/minishlab_potion-retrieval-32M/`. `lite-selective.json` is the replay.
- Quality below is the official BEIR test file, both hash halves included for ArguAna, SciDocs, TREC-COVID, Quora and Touché-2020. K=0 nDCG@10 matches the 2026-10-29 eight-set lite numbers to four digits on every set.
- Quora is the predeclared sample of 200 judged queries (seed `20261029`, population 10,000, first id `100298`, last id `98114`), scored against the full 522,931-document corpus. That sample's K=0 nDCG@10 is 0.8506. The official full-split lite number is 0.8344. The sample does not replace it.
- Latency is replayed step time, not a fresh clock of a batch of exactly K pairs. `rerank_per_pair` was timed on the trace's union batch (min-max rescore heads at weights 0.2, 0.4 and 0.6, plus the configured head). Reported rerank cost is `K * rerank_per_pair` from that batch.
- Built-in `gyr bench trace` verify scores `split == "test"` only. On the hash-split sets that is one half of the BEIR test file, so those live nDCG figures (ArguAna 0.4953, SciDocs 0.1626, TREC-COVID 0.6770, Quora sample half 0.8531, Touché 0.2799) are not the table below. nDCG matched live exactly on that half for `rescoremax:0.45` on every set. Replay p50 with no rerank was within 10% of live on SciFact (1.244 vs 1.219 ms) and NFCorpus (1.119 vs 1.107 ms). It sat higher than live on the other six: ArguAna 1.650 vs 1.398, FiQA 2.296 vs 1.845, SciDocs 1.777 vs 1.602, TREC-COVID 3.400 vs 2.418, Quora 5.539 vs 4.249, Touché 4.960 vs 3.516. At K≥8 the cross-encoder term dominates that gap except on Quora, where a pair is about 1.6 ms.

## Pair time

Official-test queries. The union column is the batch the per-pair time was measured on.

| Set | Test queries | ms/pair p50 | ms/pair p99 | Union size p50 |
|---|---:|---:|---:|---:|
| SciFact | 300 | 11.27 | 16.02 | 72 |
| NFCorpus | 323 | 10.81 | 12.09 | 62 |
| ArguAna | 1406 | 10.63 | 11.70 | 65 |
| FiQA | 648 | 11.83 | 13.89 | 71 |
| SciDocs | 1000 | 11.21 | 12.26 | 68 |
| TREC-COVID | 50 | 12.96 | 15.24 | 83 |
| Quora sample | 200 | 1.57 | 3.02 | 69 |
| Touché-2020 | 49 | 12.45 | 14.22 | 69 |

The October 24 note of about 22 ms/pair, and therefore about 180 ms for the top 8, was a different measurement. On this host a long query is 11–13 ms/pair, so the top 8 is about 90 ms and the top 50 is about 530–650 ms. Quora questions are short, so the same model is 1.6 ms/pair and the top 8 is 18 ms p50.

## nDCG@10

Recall@100 is in the last column. It does not change with K: the reranker only reorders the retrieved head.

| Set | n | K=0 | K=8 | K=16 | K=20 | K=32 | K=50 | Recall@100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.6977 | 0.6869 | 0.6816 | 0.6833 | 0.6784 | 0.9282 |
| NFCorpus | 323 | 0.3435 | 0.3546 | 0.3533 | 0.3536 | 0.3520 | 0.3510 | 0.2889 |
| ArguAna | 1406 | 0.4912 | 0.5023 | 0.5002 | 0.4979 | 0.4905 | 0.4810 | 0.9680 |
| FiQA | 648 | 0.2659 | 0.2979 | 0.3177 | 0.3189 | 0.3309 | 0.3385 | 0.5514 |
| SciDocs | 1000 | 0.1646 | 0.1699 | 0.1695 | 0.1663 | 0.1665 | 0.1660 | 0.3608 |
| TREC-COVID | 50 | 0.6797 | 0.6885 | 0.7187 | 0.7232 | 0.7210 | 0.7520 | 0.1110 |
| Quora sample | 200 | 0.8506 | 0.8513 | 0.8493 | 0.8460 | 0.8456 | 0.8444 | 0.9790 |
| Touché-2020 | 49 | 0.3183 | 0.3227 | 0.3031 | 0.3096 | 0.2910 | 0.2852 | 0.5534 |

Deeper K is not uniformly better.

- FiQA keeps rising: +0.032 at K=8, +0.053 at K=20, +0.065 at K=32, +0.073 at K=50, against this configuration's K=0 of 0.2659. The older "+0.08 on FiQA" was a validation-split comparison (rescore 0.4 with rerank 32 at 0.325 versus BM25 at 0.240 in the 2026-10-24 review), not this test-set top-8 point.
- TREC-COVID keeps rising and is largest at K=50 (+0.072). Fifty queries.
- NFCorpus, ArguAna and SciDocs peak at K=8 (+0.011, +0.011, +0.005).
- SciFact is worse at every K, and worse as K grows: −0.011 at K=8, −0.030 at K=50.
- Touché-2020 gains +0.004 at K=8 and is worse after that: −0.015 at K=16, −0.033 at K=50.
- ArguAna at K=32 is already under K=0 (−0.001) and at K=50 is −0.010.
- The Quora sample is flat at K=8 (+0.001) and slightly worse at K=50 (−0.006). It is not the full 10,000-query split.

## Latency (replayed milliseconds)

p50:

| Set | K=0 | K=8 | K=16 | K=20 | K=32 | K=50 |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 1.2 | 91.5 | 181.6 | 226.7 | 362.0 | 564.8 |
| NFCorpus | 1.1 | 87.6 | 174.1 | 217.4 | 347.1 | 541.8 |
| ArguAna | 1.7 | 86.7 | 171.8 | 214.2 | 341.8 | 533.1 |
| FiQA | 2.3 | 97.1 | 191.7 | 239.0 | 381.1 | 594.1 |
| SciDocs | 1.8 | 91.5 | 181.2 | 226.0 | 360.5 | 562.3 |
| TREC-COVID | 3.4 | 107.6 | 211.2 | 263.0 | 418.5 | 651.7 |
| Quora sample | 5.5 | 18.0 | 30.6 | 37.0 | 55.7 | 84.0 |
| Touché-2020 | 5.0 | 104.4 | 203.8 | 253.5 | 403.1 | 627.5 |

p99:

| Set | K=0 | K=8 | K=16 | K=20 | K=32 | K=50 |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 1.6 | 129.4 | 257.6 | 321.7 | 513.9 | 802.2 |
| NFCorpus | 1.2 | 97.9 | 194.6 | 242.9 | 388.0 | 605.6 |
| ArguAna | 2.1 | 95.3 | 188.9 | 235.6 | 376.0 | 586.6 |
| FiQA | 2.8 | 113.4 | 224.5 | 280.1 | 446.7 | 696.7 |
| SciDocs | 2.0 | 99.7 | 197.8 | 246.8 | 393.9 | 614.5 |
| TREC-COVID | 4.5 | 125.0 | 246.8 | 307.8 | 490.6 | 764.9 |
| Quora sample | 7.0 | 29.6 | 53.8 | 65.8 | 102.1 | 156.4 |
| Touché-2020 | 6.8 | 118.6 | 232.3 | 289.1 | 459.7 | 715.6 |

FiQA K=8 p99 is 113 ms on this host. The older ~180 ms figure is the 22 ms/pair estimate, not this run.

## Selective rerank (replay only)

Protocol, locked before looking at these test scores. Two separate rules:

- Overlap: rerank when the Jaccard of the BM25 top 10 and the dense top 10 is below T. Grid 0.0 to 1.1 step 0.1.
- Margin: rerank when the configured fusion's rank-1 score minus its rank-2 score is below T. Grid is 0 and 21 steps across the tuning-validation margins, plus one step past the max.

T is fit only on the tuning validation splits (SciFact train, NFCorpus dev, ArguAna and FiQA hash-validation; 2,323 queries). Predeclared band: pooled validation nDCG@10 within 0.005 of always reranking at the same K. Among thresholds inside the band, smallest total rerank milliseconds wins. Held-out queries do not move T. Both rules stayed inside the band at every K.

The margin threshold came out the same at every K (0.0379). That is what the validation objective selected, not a constraint.

| K | Rule | T | Val nDCG sel | Val nDCG always | Drop | Queries reranked | Rerank ms | Always rerank ms | Mean ms | Always mean ms | p50 ms | p99 ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | margin | 0.0379 | 0.5065 | 0.5054 | −0.0012 | 549/2323 | 49,492 | 204,464 | 22.9 | 89.6 | 1.65 | 102.7 |
| 8 | overlap | 0.20 | 0.5028 | 0.5054 | +0.0026 | 958/2323 | 86,495 | 204,464 | 38.8 | 89.6 | 1.86 | 103.8 |
| 16 | margin | 0.0379 | 0.5087 | 0.5089 | +0.0002 | 549/2323 | 98,985 | 408,927 | 44.2 | 177.6 | 1.65 | 202.9 |
| 16 | overlap | 0.20 | 0.5065 | 0.5089 | +0.0025 | 958/2323 | 172,990 | 408,927 | 76.1 | 177.6 | 1.86 | 205.4 |
| 20 | margin | 0.0379 | 0.5093 | 0.5082 | −0.0011 | 549/2323 | 123,731 | 511,159 | 54.9 | 221.6 | 1.65 | 253.1 |
| 20 | overlap | 0.20 | 0.5077 | 0.5082 | +0.0005 | 958/2323 | 216,237 | 511,159 | 94.7 | 221.6 | 1.86 | 256.1 |
| 32 | margin | 0.0379 | 0.5093 | 0.5051 | −0.0042 | 549/2323 | 197,970 | 817,854 | 86.8 | 353.7 | 1.65 | 403.6 |
| 32 | overlap | 0.10 | 0.5001 | 0.5051 | +0.0050 | 338/2323 | 123,451 | 817,854 | 54.7 | 353.7 | 1.58 | 401.7 |
| 50 | margin | 0.0379 | 0.5095 | 0.5041 | −0.0054 | 549/2323 | 309,327 | 1,277,898 | 134.7 | 551.7 | 1.65 | 629.7 |
| 50 | overlap | 0.10 | 0.5002 | 0.5041 | +0.0039 | 338/2323 | 192,893 | 1,277,898 | 84.6 | 551.7 | 1.58 | 626.3 |

On the validation split the margin rule reranks 24% of queries and keeps 24% of the always-rerank milliseconds at K=8 (76% of rerank time saved) with pooled nDCG 0.001 above always-rerank. Overlap reranks 41% and keeps 42% of the rerank time, 0.003 under always-rerank. p50 falls to the no-rerank latency because most queries are skipped. p99 stays on the rerank path. Selective rerank does not cut the tail.

A negative drop means the selective pool scored higher than always-rerank. At K=32 and K=50 that happens because always-rerank is already hurting SciFact, and the margin rule skips most of those queries. The band still holds. It is not a reason to retune T on the test sets.

Tuning-test transfer uses `split == "test"` only, the half the fit did not see. For SciFact, NFCorpus and FiQA that is the official test. For ArguAna it is the odd-sha256 half (716 of 1,406), so those cells are not comparable to the 1,406-query sweep. Held-out cells are the whole BEIR test file (Quora is still the 200-query sample).

Margin rule, K=8. nDCG@10 and how many queries were reranked:

| Set | n | Never | Selective | Always | Reranked |
|---|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.7173 | 0.6977 | 68 |
| NFCorpus | 323 | 0.3435 | 0.3511 | 0.3546 | 116 |
| ArguAna (hash half) | 716 | 0.4953 | 0.4950 | 0.5109 | 11 |
| FiQA | 648 | 0.2659 | 0.2894 | 0.2979 | 331 |
| SciDocs | 1000 | 0.1646 | 0.1679 | 0.1699 | 439 |
| TREC-COVID | 50 | 0.6797 | 0.6871 | 0.6885 | 34 |
| Quora sample | 200 | 0.8506 | 0.8572 | 0.8513 | 48 |
| Touché-2020 | 49 | 0.3183 | 0.3229 | 0.3227 | 37 |

Held-out rerank milliseconds at this point: 47,212 selective versus 102,647 always (54% of rerank time saved) across 558 of 1,299 queries.

What got worse against always-rerank at K=8, margin: ArguAna's hash half gives up the gain (0.4950 versus 0.5109, only 11 queries reranked). FiQA keeps 0.023 of the 0.032 gain. NFCorpus and SciDocs each sit about 0.002 to 0.003 under always-rerank and still above no rerank. SciFact selective (0.7173) beats both always-rerank (0.6977) and no rerank (0.7088), because the rule skips queries where MiniLM hurts.

Margin rule, K=50:

| Set | n | Never | Selective | Always | Reranked |
|---|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.7203 | 0.6784 | 68 |
| NFCorpus | 323 | 0.3435 | 0.3464 | 0.3510 | 116 |
| ArguAna (hash half) | 716 | 0.4953 | 0.4946 | 0.4871 | 11 |
| FiQA | 648 | 0.2659 | 0.3125 | 0.3385 | 331 |
| SciDocs | 1000 | 0.1646 | 0.1644 | 0.1660 | 439 |
| TREC-COVID | 50 | 0.6797 | 0.7176 | 0.7520 | 34 |
| Quora sample | 200 | 0.8506 | 0.8552 | 0.8444 | 48 |
| Touché-2020 | 49 | 0.3183 | 0.2812 | 0.2852 | 37 |

At K=50 the same frozen threshold gives up 0.026 of FiQA's gain versus always-rerank (0.3125 versus 0.3385) and 0.034 of TREC-COVID's (0.7176 versus 0.7520). SciDocs selective falls 0.0002 under no rerank. Touché selective is 0.037 under no rerank; always-rerank is 0.033 under no rerank. SciFact selective stays above no rerank while always-rerank is 0.030 under it. K=16, K=20 and K=32 are in `lite-selective.json` under each rule's `tuning_test` and `heldout_test`.

## Full tier

Not measured. Indexing bge-small-en-v1.5 at about 25 documents/s on 4 cores is roughly 40 minutes for FiQA, 2 hours for TREC-COVID, 4 hours for Touché-2020 and 6 hours for Quora before the cross-encoder runs. Open RAG Bench was not part of this sweep either.

## Reproduce

```
gyr bench trace --dataset <name> --rerank-dir <MiniLM> --rerank-k 50 --depth 100 \
  --tier lite --models <models> --out-dir bench/results/2026-10-29-rerank-sweep/lite
python3 bench/rerank_selective.py lite/minishlab_potion-retrieval-32M/*.jsonl \
  --tier lite --out lite-selective.json
```

Quora's dataset directory must be named `quora` and its `qrels/test.tsv` filtered to the 200 ids in `bench/results/2026-10-29-heldout-frameworks/quora-sample.json`. The corpus stays complete.
