# Lite tier, eight BEIR sets, configured rule and RRF

`gyr bench beir --tier lite --fusions configured,rrf` at commit 61a4293, one run, 4 vCPUs
(Intel Xeon), test split. `configured` is the lite tier's rule (`rescore-max`, dense weight
0.45). `rrf` is reciprocal rank fusion with weights 1 and 1, the rule LlamaIndex and Haystack
use. Both rules are scored on the same index. Documents are indexed as one section each, the
same way as the published head-to-head.

nDCG@10 for `configured` matches the published lite table (`2026-10-26-quality-sweep`, commit
90f1580) to four digits on every set. The static embedder is deterministic, so that match is
the check that this run is the same engine.

| Dataset | Docs | Index s | nDCG@10 configured | nDCG@10 RRF | Recall@100 configured | RRF | Query p50 ms configured | RRF |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SciFact | 5,183 | 0.24 | 0.7088 | 0.6978 | 0.9282 | 0.9587 | 1.24 | 1.11 |
| NFCorpus | 3,633 | 0.18 | 0.3435 | 0.3395 | 0.2889 | 0.3068 | 1.17 | 1.03 |
| ArguAna | 8,674 | 0.28 | 0.4912 | 0.4893 | 0.9680 | 0.9730 | 1.42 | 1.29 |
| FiQA | 57,638 | 1.11 | 0.2659 | 0.2432 | 0.5514 | 0.5621 | 1.84 | 1.67 |
| TREC-COVID | 171,332 | 7.90 | 0.6797 | 0.6312 | 0.1110 | 0.1018 | 2.63 | 2.26 |
| SciDocs | 25,657 | 0.68 | 0.1646 | 0.1585 | 0.3608 | 0.3790 | 1.59 | 1.44 |
| Quora | 522,931 | 3.84 | 0.8344 | 0.8178 | 0.9781 | 0.9860 | 4.11 | 3.96 |
| Touché-2020 | 382,545 | 29.85 | 0.3183 | 0.3089 | 0.5534 | 0.5597 | 3.80 | 3.32 |

Index time and query latency are a single run on this machine. They are not the five-run
medians in `2026-10-26-lite-variance`; this CPU is faster than the 2.10 GHz host those used.
Quality does not move between runs.
