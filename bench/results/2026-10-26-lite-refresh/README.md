# Gyrfalcon lite cells refreshed, 2026-10-26

`bench/compare/compare.py --systems gyrfalcon --cores 4,1 --datasets scifact,nfcorpus,arguana,fiqa`
at commit 621dc84: the same cells as `2026-10-25-cloud-4core-h2h` and `2026-10-25-cloud-1core-h2h`
after the mapped embeddings table, compact chunk records, and length-sorted embedding batches.
Scores are unchanged; the other frameworks' rows in those directories still apply.

| cores | dataset | nDCG@10 | index s (before) | query p50 ms (before) | peak RSS MB (before) |
|---|---|---|---|---|---|
| 4 | scifact | 0.7092 | 0.32 (0.46) | 1.70 (1.91) | 268 (303) |
| 4 | nfcorpus | 0.3410 | 0.28 (0.31) | 1.60 (1.76) | 258 (294) |
| 4 | arguana | 0.4729 | 0.44 (0.58) | 2.08 (2.45) | 282 (344) |
| 4 | fiqa | 0.2689 | 2.00 (2.65) | 2.68 (2.95) | 414 (520) |
| 1 | scifact | 0.7092 | 0.92 (1.14) | 2.18 (1.98) | 224 (295) |
| 1 | nfcorpus | 0.3410 | 0.73 (0.81) | 1.74 (1.79) | 206 (294) |
| 1 | arguana | 0.4729 | 1.38 | 2.72 | 259 |
| 1 | fiqa | 0.2689 | 6.14 | 3.74 | 352 |

Peak RSS includes the harness's own copy of the corpus.
