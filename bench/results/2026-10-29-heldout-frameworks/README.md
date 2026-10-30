# Same-rule comparison on all eight BEIR test sets

nDCG@10. Lite tier, potion-retrieval-32M, depth 100, 4 vCPU. Gyrfalcon `configured` is
rescore-max with dense weight 0.45. Gyrfalcon `rrf` and the three frameworks fuse BM25 and dense
with reciprocal rank fusion. The corpus is always the full dataset. No weight was chosen on these
runs.

The four tuning-set framework cells are copied from `bench/results/2026-10-25-cloud-4core-h2h`
and were not re-run. The Gyrfalcon cells are the full test splits from
`bench/results/2026-10-29-lite-8set` (commit 61a4293), except the Quora sample row, which is scored
from `gyrfalcon-quora-runs/` on the ids in `quora-sample.json`. Held-out framework cells are this
directory.

| Dataset | Queries | Gyrfalcon configured | Gyrfalcon RRF | LlamaIndex | LangChain | Haystack |
|---|---:|---:|---:|---:|---:|---:|
| SciFact | 300 | 0.7088 | 0.6978 | 0.6945 | 0.6040 | 0.6739 |
| NFCorpus | 323 | 0.3435 | 0.3395 | 0.3081 | 0.2543 | 0.3265 |
| ArguAna | 1,406 | 0.4912 | 0.4893 | 0.4725 | 0.3867 | 0.4414 |
| FiQA | 648 | 0.2659 | 0.2432 | 0.2420 | 0.1883 | 0.2388 |
| TREC-COVID | 50 | 0.6797 | 0.6312 | 0.6355 | 0.5433 | 0.6016 |
| SciDocs | 1,000 | 0.1646 | 0.1585 | 0.1571 | 0.1270 | 0.1563 |
| Quora | 10,000 | 0.8344 | 0.8178 |  |  |  |
| Quora sample | 200 of 10,000 | 0.8506 | 0.8292 | 0.7900 | 0.7244 | 0.7961 |
| Touché-2020 | 49 | 0.3183 | 0.3089 | 0.2816 | 0.2815 | **0.3337** |

Empty framework cells on the full Quora row were not run. The sample row is the comparison on
one shared list of 200 judged queries.

## What is worse

- Gyrfalcon RRF is below LlamaIndex on TREC-COVID (0.6312 vs 0.6355). Configured still leads
  that set (0.6797).
- Haystack is ahead of both Gyrfalcon rules on Touché-2020 (0.3337 vs 0.3183 configured and
  0.3089 RRF). Its recall@100 on that set is 0.5766, against 0.5534 configured and 0.5597 RRF.
- On the Quora sample, configured 0.8506 is higher than the official full-split 0.8344. The
  sample does not replace the full-split number.

## How this run was done

Held-out sets are SciDocs, TREC-COVID, Touché-2020, and Quora. SciDocs, TREC-COVID, and
Touché-2020 use every judged test query. Quora uses 200 queries drawn before any of these scores
were computed: `random.Random(20261029).sample` of the sorted judged list, then sorted by id.
Population 10,000. The ids are stored in `quora-sample.json` (first `100298`, last `98114`).
Do not re-derive them from the seed in another Python.

This machine is Python 3.12.3. The 2026-10-25 tuning cells were Python 3.13. Pins that match
both writes of the harness: llama-index-core 0.14.25, langchain-core 1.7.0, langchain-classic
1.0.8, haystack-ai 3.4.1, fastembed 0.9.1, model2vec 0.10.0. This run also resolved
langchain-community 0.4.2, llama-index-retrievers-bm25 0.8.0, rank-bm25 0.2.2, bm25s 0.3.16,
numpy 2.5.4 (`requirements-freeze.txt`). The tuning cells were not rebuilt, so their transitive
versions are whatever that earlier machine resolved.

Threads were pinned with `taskset -c 0-3` and `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`,
`MKL_NUM_THREADS`, and `RAYON_NUM_THREADS` set to 4. One process at a time. Peak RSS is
`ru_maxrss` of the child in the `.measure.json` files.

TREC-COVID's corpus has one empty document. LlamaIndex logged `Some nodes are missing content,
skipping them...` once.

## Latency on this machine

nDCG does not depend on it. The clocks do.

SciDocs fits comfortably (peak RSS 1.2 to 1.6 GB). TREC-COVID peaks at 5.7 to 7.9 GB. Touché-2020
and Quora peak at 13.1 to 14.8 GB on a 15 GB machine, and the run used a 16 GB swap file. LangChain
on Touché-2020 swapped about 8 GB; its query p50 is 29.5 s. Quora LangChain p50 is 16.9 s,
Haystack 10.8 s, LlamaIndex 7.5 s. Those figures are not comparable to Gyrfalcon's full-split
Quora p50 of about 4 ms, and they are not comparable to a run that stayed in RAM.

Gyrfalcon's own Quora sample was not re-timed. `gyrfalcon-quora.json` is the full 10,000-query run.

## Still unknown

The three frameworks were not run on the full 10,000 Quora queries. A tuned setup for each
framework (stemming, a faster BM25 backend) is not in this table; the cells are library defaults,
same as the 2026-10-25 run.
