# Quality without speed, 2026-10-26

Goal: higher nDCG@10 on every dataset at the current query and ingest speed. Protocol: choices
are made on the tuning sets (SciFact, NFCorpus, ArguAna, FiQA, ORB; validation splits where
they exist) and reported on their test splits plus four held-out BEIR sets that no setting was
ever chosen on (TREC-COVID, SciDocs, Quora, Touché-2020). Lite tier, 4-core VM, `gyr bench beir`.
Fusion variants were prototyped offline on planner traces, which reproduce live nDCG exactly.

## Kept

| lite, nDCG@10 | scifact | nfcorpus | arguana | fiqa | trec-covid | scidocs | quora | touche | tuning | held-out |
|---|---|---|---|---|---|---|---|---|---|---|
| before (rescore 0.4) | 0.7092 | 0.3410 | 0.4729 | 0.2689 | 0.6441 | 0.1643 | 0.8319 | 0.3284 | 0.4480 | 0.4922 |
| after | 0.7088 | 0.3435 | 0.4912 | 0.2659 | 0.6797 | 0.1646 | 0.8344 | 0.3183 | 0.4524 | 0.4993 |
| query p50 before / after, ms | 1.9 / 1.8 | 1.7 / 1.5 | 2.3 / 2.1 | 3.1 / 2.7 | 3.7 / 3.4 | 2.3 / 2.4 | 6.0 / 6.3 | 5.2 / 5.9 | | |

- **Max fusion for the lite tier** (`Fusion::RescoreMax`, dense weight 0.45): BM25 divided by
  the query's best BM25 score, plus raw cosine, instead of min-max stretching both. Min-max
  sends the weakest lexical candidate to zero however good it is. Held-out +0.008 (TREC-COVID
  +0.037 and Quora +0.005 significant), tuning +0.0015. For the full tier it is worse at every
  weight (bge's cosines sit in a narrow band and need stretching), so the full tier keeps
  min-max rescore at 0.65.
- **Repeated query terms count**: a term the query holds n times weighs n, as a repeated clause
  does in Lucene and Tantivy. ArguAna BM25 0.403 to 0.482 and fused +0.014; other sets within
  ±0.003. Full tier ArguAna about 0.602 to 0.612; SciFact 0.7431 to 0.7430, NFCorpus 0.3620 to
  0.3608.

Index time is unchanged; query latency moves within run-to-run noise.

## Tried and dropped

| idea | result |
|---|---|
| Dense pseudo-relevance feedback (top-k fused vectors into the query, second dense pass) | held-out +0.001 to +0.002 over max fusion; query time up to 2x on the large sets (Quora 6 to 11 ms) |
| BM25 k1 and b | dataset-dependent: b 0.4 takes Touché BM25 0.333 to 0.447 but ArguAna 0.403 to 0.316; k1 2.0 helps BM25 alone (+0.008) but not after fusion |
| Window max-sim for static embeddings (best 64-word window blended with the whole document) | dense alone up to +0.035 (SciFact), after fusion +0.002 to +0.005 on four sets and -0.006 to -0.012 on TREC-COVID |
| Titles repeated in the indexed text | SciFact +0.007, SciDocs +0.006; Touché -0.040, TREC-COVID -0.010, ArguAna -0.008 |
| Per-query fusion weight from query length or leg overlap | the validation choice was one weight for every bucket |
| `static-retrieval-mrl-en-v1` instead of potion | lower on three of four BEIR sets (spec, "Lite model candidates") |

Within the current speed the remaining gains are small: the lite tier is bound by its static
dense leg. Larger steps need a stronger embedding or a reranker, which cost latency.
