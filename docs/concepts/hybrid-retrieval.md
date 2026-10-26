<sub>[Docs](../README.md) / Concepts</sub>

# Hybrid retrieval

Every Gyrfalcon query runs two searches over the same chunks and combines them. This page
explains what each search is good at, how the two are combined, and how much each part
contributes on public benchmarks.

## Why two legs

| | BM25 (lexical) | Dense (vectors) |
|---|---|---|
| Matches | the words of the query, weighted by rarity | the meaning of the query, as the model learned it |
| Strong on | names, codes, rare terms, exact phrases, identifiers such as `E0308` | paraphrases, synonyms, questions phrased unlike the answer |
| Weak on | vocabulary mismatch ("car" vs "automobile") | rare terms the model never saw, exact identifiers |
| Cost per query | scales with the postings of the query's terms | scales with the number of chunks |

Neither leg wins everywhere. On the eight BEIR sets below, BM25 alone beats the lite tier's
static dense leg on all eight, yet fusing the two beats both legs on seven. With the full tier's
transformer the dense leg is the stronger one (SciFact 0.711 against BM25's 0.686), and fusion
still adds to it.

Lite tier, BEIR test splits, nDCG@10 (`bench/results/2026-10-26-quality-sweep/lite-after.json`):

| Dataset | BM25 only | Dense only | Fused |
|---|---|---|---|
| SciFact | 0.684 | 0.632 | **0.709** |
| NFCorpus | 0.323 | 0.308 | **0.344** |
| ArguAna | 0.482 | 0.452 | **0.491** |
| FiQA | 0.250 | 0.189 | **0.266** |
| TREC-COVID | 0.604 | 0.440 | **0.680** |
| SciDocs | 0.156 | 0.138 | **0.165** |
| Quora | 0.809 | 0.784 | **0.834** |
| Touché-2020 | **0.337** | 0.211 | 0.318 |

Touché-2020 is the one set where fusion lost to BM25 alone; its relevant documents are long
arguments that share many words with the query, and the static model's vectors add noise there.

## The lexical leg

Gyrfalcon owns its BM25 index rather than wrapping a search library. Its scores match
Tantivy's (and Lucene's) BM25 with `k1 = 1.2` and `b = 0.75`, including the one-byte length
norms; on 2,677 BEIR test queries every top-100 list was identical to Tantivy's.

For a query term `t` and a chunk `d`:

```text
idf(t)      = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))
score(t, d) = idf(t) * tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * len(d) / avglen))
BM25(q, d)  = sum over the query's distinct terms t of  count(t, q) * score(t, d)
```

`N`, `df`, and `avglen` are collection-wide: summed over every segment and the write buffer,
so a chunk's score does not depend on which segment holds it. `count(t, q)` is how many times
the term appears in the query; see [design record 0003](../design/0003-query-term-frequency.md).

Terms come from the `gyr-text-v2` analyzer: split on non-alphanumerics, lowercase, drop 33
English stopwords, Snowball-stem ASCII words. Stemming and stopwords raised BM25 nDCG@10 by
0.01 to 0.025 on three of four BEIR sets over plain lowercased tokens (spec, "Analyzer
`gyr-text-v2`").

## The dense leg

The query is embedded once by the tier's model and compared by cosine similarity with every
live chunk, through a three-stage cascade:

1. **Prefilter.** Each chunk's vector is centered on its segment's mean, rotated by a seeded
   random Hadamard transform, and reduced to its sign bits. Hamming distance between the query's
   bits and every chunk's bits costs about 3 ns per chunk and keeps the best 4,000.
2. **Estimate.** For those, a 4-bit version of the query gives an estimate of the cosine (the
   RaBitQ estimator).
3. **Exact.** The best 800 estimates are rescored with the exact int8 cosine.

This finds 0.996 to 1.000 of the exact top 50, higher than the HNSW graph it replaced (0.94 to
0.985), and needs no graph build: a segment's codes are written in one pass.

## Fusion

The two legs each return their top 50. Fusion turns them into one list.

### The rules

Let `w` be the dense weight, `bm25(d)` the chunk's BM25 score (0 if BM25 did not return it), and
`cos(d)` its cosine similarity.

| Rule | Score of a candidate | Used by |
|---|---|---|
| `rrf` | `1 / (60 + rank_bm25) + 1 / (60 + rank_dense)` | available, not a default |
| `score` | `(1 - w) * minmax(bm25) + w * minmax(cos)`, each leg's own candidates only | available |
| `rescore` | as `score`, but every candidate from either leg gets its exact `cos` | full tier, `w = 0.65` |
| `rescore-max` | `(1 - w) * bm25 / max(bm25) + w * cos`, with exact `cos` for every candidate | lite tier, `w = 0.45` |

`minmax(x)` maps the best candidate to 1 and the worst to 0. `max(bm25)` is the best BM25
score among the query's candidates.

### Why rescore instead of rank fusion

Reciprocal rank fusion keeps only the order of each list and throws away how far apart the
scores are. When one leg is much weaker than the other, its noise gets the same say as the
strong leg's signal. On Open RAG Bench with the lite model, RRF ranked below BM25 alone (MRR@10
0.563 against 0.740) while rescore reached 0.752 (spec, "v2 changes").

Rescoring also fixes a second problem. A chunk that only BM25 found has no dense score in the
`score` rule and is treated as the worst on that side, even when its true cosine is high. In
`rescore` and `rescore-max` every candidate's cosine is computed from its stored vector, so a
lexical-only find is judged on its merit. That costs one int8 dot product per candidate, about
0.2 to 0.4 ms per query.

### Why the two tiers fuse differently

Min-max stretching maps the weakest BM25 candidate to zero however good it is. With the lite
tier's static embeddings, cosines spread widely and can be used as they are, so the lite rule
keeps BM25 relative to the query's best match and adds the raw cosine. That raised the held-out
mean nDCG@10 by 0.008 (TREC-COVID +0.037 and Quora +0.005, both significant).

bge-small's cosines sit in a narrow band: unrelated English texts still score well above zero.
Used raw they barely separate candidates, so the full tier stretches both sides with
min-max. Max fusion lost to min-max for the full tier at every weight tried.

See [design record 0001](../design/0001-hybrid-fusion.md) for the full comparison.

### How the weights were chosen

Each weight was chosen on validation queries only, then confirmed once on test queries and, for
the lite rule, on four BEIR sets that were never used to choose anything. The method, and why it
matters for trusting the numbers, is in [Planner](planner.md) and
[Methodology](../benchmarks/methodology.md).

| Tier | Rule | Weight | Validation mean nDCG@10 around the choice |
|---|---|---|---|
| lite | `rescore-max` | 0.45 | best mean over the five tuning sets' validation splits |
| full | `rescore` | 0.65 | 0.6345 at 0.55, 0.6374 at 0.6, **0.6392 at 0.65**, 0.6379 at 0.7 |

### A worked example

Three candidates for one query on the lite tier (`w = 0.45`). The numbers are illustrative.

| Chunk | BM25 | `bm25 / max` | Cosine | Fused: `0.55 * bm25/max + 0.45 * cos` |
|---|---|---|---|---|
| A, found by both legs | 12.0 | 1.00 | 0.52 | **0.784** |
| B, found by BM25 only | 9.0 | 0.75 | 0.61 | 0.687 |
| C, found by dense only | 0 | 0.00 | 0.70 | 0.315 |

Under `score` (no rescoring), B would have no cosine and would rank on BM25 alone; with min-max,
the weakest BM25 candidate would drop to 0 regardless of its score. `rescore-max` keeps B's
strong cosine and keeps BM25 proportional.

## Choosing a rule yourself

The defaults are the tier's rule. The command line can override them per run:

```sh
gyr query ./kb "rust borrow checker error" --fusion rrf
gyr query ./kb "rust borrow checker error" --fusion rescore --dense-alpha 0.3
```

In Rust, set `index.fusion` to any `Fusion` variant. The Python package always uses the tier's
rule. To compare rules on your own judged queries, `gyr bench beir --fusions` scores several on
one index in one run; see [Reproducing](../benchmarks/reproducing.md).

---

<sub>Previous: [Architecture](architecture.md) · Next: [Tiers](tiers.md)</sub>
