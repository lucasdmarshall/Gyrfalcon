# Baseline (P0), 2026-10-24

4-core shared cloud VM (Intel Xeon @ 2.10GHz). Planner work compares against these files.

| File | Command |
|---|---|
| `beir-lite.json` | `gyr bench beir --dataset .data/beir/{scifact,nfcorpus,fiqa,arguana} --fusions lexical,dense,rrf,rescore:0.2,...,rescore:0.6` |
| `beir-full.json` | same with `--tier full` and `rescore:0.7` added |
| `gyrfalcon-*.json` | `gyr bench accuracy` on Open RAG Bench text-only (see the v2 plan) |

These are full test-split numbers, measured before the validation/test split of the planner
protocol existed. They are a reference, not tuning data: planner choices are made on the
validation splits only.

nDCG@10 on BEIR test:

| Tier | Dataset | BM25 | dense | RRF | rescore 0.2 | 0.3 | 0.4 | 0.5 | 0.6 | 0.7 |
|---|---|---|---|---|---|---|---|---|---|---|
| lite | SciFact | 0.661 | 0.622 | 0.664 | 0.682 | 0.688 | 0.693 | 0.696 | 0.697 | |
| lite | NFCorpus | 0.309 | 0.302 | 0.324 | 0.322 | 0.327 | 0.330 | 0.334 | 0.336 | |
| lite | FiQA | 0.237 | 0.186 | 0.233 | 0.242 | 0.245 | 0.251 | 0.255 | 0.257 | |
| lite | ArguAna | 0.405 | 0.450 | 0.462 | 0.458 | 0.467 | 0.470 | 0.471 | 0.468 | |
| full | SciFact | 0.661 | 0.712 | 0.712 | 0.690 | 0.696 | 0.708 | 0.715 | 0.729 | 0.743 |
| full | NFCorpus | 0.308 | 0.337 | 0.350 | 0.328 | 0.334 | 0.340 | 0.346 | 0.348 | 0.350 |
| full | FiQA | 0.237 | 0.377 | 0.345 | 0.257 | 0.269 | 0.284 | 0.304 | 0.331 | 0.363 |
| full | ArguAna | 0.405 | 0.592 | 0.539 | 0.519 | 0.552 | 0.578 | 0.592 | 0.596 | 0.602 |

Index time: lite 2–21 s per dataset; full 267–4,815 s (FiQA, 57,638 docs).
