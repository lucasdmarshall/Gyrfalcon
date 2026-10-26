# Lite tier run-to-run variance, 2026-10-26

Five back-to-back runs of `gyr bench beir` over eight BEIR sets at commit d3dd62b (engine identical
to 90f1580, the quality sweep's commit), lite tier, default fusion, 4-vCPU cloud VM (Intel Xeon @
2.10 GHz). Nothing else ran on the machine. Each run builds every index from scratch and times every
test query once. Summarized with `python3 bench/summarize_runs.py run*.json`; rerun with
`scripts/reproduce.sh lite`.

| Dataset | Documents | Queries | nDCG@10 | Index s, median (range) | Query p50 ms, median (range) | Query p99 ms, median (range) |
|---|---|---|---|---|---|---|
| scifact | 5,183 | 300 | 0.7088 | 0.37 (0.32 to 0.42) | 1.77 (1.72 to 1.85) | 4.06 (2.65 to 5.40) |
| nfcorpus | 3,633 | 323 | 0.3435 | 0.28 (0.27 to 0.31) | 1.52 (1.47 to 1.70) | 2.63 (2.10 to 4.58) |
| arguana | 8,674 | 1,406 | 0.4912 | 0.43 (0.39 to 0.58) | 2.05 (1.96 to 2.30) | 4.38 (3.67 to 5.37) |
| fiqa | 57,638 | 648 | 0.2659 | 1.81 (1.70 to 2.02) | 2.66 (2.45 to 2.76) | 4.48 (3.87 to 4.90) |
| trec-covid | 171,332 | 50 | 0.6797 | 12.97 (11.47 to 14.15) | 3.50 (2.99 to 3.73) | 84.34 (82.29 to 96.74) |
| scidocs | 25,657 | 1,000 | 0.1646 | 1.10 (1.06 to 1.21) | 2.26 (2.15 to 2.28) | 4.39 (4.13 to 4.87) |
| quora | 522,931 | 10,000 | 0.8344 | 6.81 (6.60 to 8.00) | 5.70 (5.64 to 6.02) | 9.81 (9.50 to 10.80) |
| webis-touche2020 | 382,545 | 49 | 0.3183 | 44.09 (41.95 to 45.30) | 5.22 (5.00 to 7.17) | 237.29 (207.74 to 409.84) |

- nDCG@10 was identical to every digit in all five runs: the static embedder and every stage of the
  lite pipeline are deterministic.
- The largest p50 spread is 1.4x (Touché, one run at 7.17 ms); on the other seven sets it is at
  most 1.25x. Index time varies by up to 1.5x on the
  smallest sets, where it is under half a second.
- p99 on TREC-COVID and Touché-2020 (about 50 queries each) is in effect the slowest query. The
  outlier appears only in the first fusion measured on a freshly built index: in
  `2026-10-26-quality-sweep/lite-before.json`, the same queries measured afterwards on the same
  index as lexical, dense, and RRF runs have p99 of 5 to 13 ms. The cause is not isolated yet.
