# Lite tier run-to-run variance, 2026-10-26

Five back-to-back runs of `gyr bench beir` over eight BEIR sets at commit de49193, lite tier,
default fusion, 4-vCPU cloud VM (Intel Xeon @ 2.10 GHz). Nothing else ran on the machine. Each run
builds every index from scratch and times every test query once. Produced by
`scripts/reproduce.sh lite 5`, which also prints this table (`bench/summarize_runs.py`).

| Dataset | Documents | Queries | nDCG@10 | Index s, median (range) | Query p50 ms, median (range) | Query p99 ms, median (range) |
|---|---|---|---|---|---|---|
| scifact | 5,183 | 300 | 0.7088 | 0.34 (0.30 to 0.42) | 1.62 (1.53 to 1.75) | 2.50 (2.01 to 2.84) |
| nfcorpus | 3,633 | 323 | 0.3435 | 0.28 (0.26 to 0.35) | 1.59 (1.48 to 1.98) | 3.26 (2.30 to 3.84) |
| arguana | 8,674 | 1,406 | 0.4912 | 0.43 (0.41 to 0.53) | 2.00 (1.87 to 2.12) | 3.90 (3.31 to 5.00) |
| fiqa | 57,638 | 648 | 0.2659 | 1.74 (1.61 to 1.98) | 2.54 (2.40 to 2.62) | 4.02 (3.66 to 6.35) |
| trec-covid | 171,332 | 50 | 0.6797 | 12.30 (12.06 to 12.51) | 3.83 (3.17 to 4.43) | 6.77 (5.25 to 11.57) |
| scidocs | 25,657 | 1,000 | 0.1646 | 1.16 (1.11 to 1.19) | 2.20 (2.10 to 2.25) | 4.02 (3.56 to 4.49) |
| quora | 522,931 | 10,000 | 0.8344 | 6.50 (6.12 to 6.98) | 5.39 (5.37 to 5.47) | 8.97 (8.68 to 9.44) |
| webis-touche2020 | 382,545 | 49 | 0.3183 | 45.93 (45.00 to 48.11) | 5.58 (4.96 to 7.23) | 11.61 (9.24 to 20.03) |

- nDCG@10 was identical to every digit in all five runs: the static embedder and every stage of the
  lite pipeline are deterministic.
- p50 varied by at most 1.15x on five sets and at most 1.5x on NFCorpus, TREC-COVID, and Touché-2020.
- The same five runs at commit d3dd62b, before the fix in de49193, measured p99 of 84 ms on
  TREC-COVID and 237 ms on Touché-2020, with everything else within noise of this table. After a
  merge, the first query to finish deleted the merged-away segment directories on its own thread
  while holding the reaper lock that `acquire` waits on. Deletes now run on a background thread
  outside that lock (spec, "Rerank off by default, atomic HTTP update, deletes off the query
  path").
