<sub>[Docs](../README.md) / Design records</sub>

# 0001. Fuse by rescoring every candidate, with a rule per tier

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-10-24; lite rule amended 2026-10-26 |
| Code | `crates/gyrfalcon/src/fuse.rs`, `retrieve.rs`, `tier.rs` |
| Evidence | `bench/results/baseline/`, `2026-10-26-full-tier-v2/`, `2026-10-26-quality-sweep/` |

## Context

Every query produces two ranked lists, one from BM25 and one from the dense index, each with
its own scores on its own scale. They must become one list.

The first design used reciprocal rank fusion (RRF), the common default in RAG frameworks: each
document scores `1 / (60 + rank)` in each list, summed. RRF needs no tuning and no score
calibration. It also discards every score, keeping only positions.

Two observations on the first benchmarks made that a problem:

1. **A weak leg drags RRF down.** With the lite tier's static embeddings on Open RAG Bench, RRF
   ranked below BM25 alone: MRR@10 0.563 against 0.740. Rank 3 in a noisy list counts as much as
   rank 3 in a precise one.
2. **One-sided candidates are misjudged.** A document that only BM25 returned has no dense score.
   Treating it as the dense leg's worst case penalizes exactly the documents hybrid search exists
   to rescue.

## Decision

1. **Rescore.** Take the union of both legs' candidates (50 each) and compute the exact cosine
   similarity of every candidate from its stored vector, so every candidate has both a BM25 score
   (0 if BM25 did not return it) and a true cosine.
2. **Mix scores, not ranks:** `(1 - w) * lexical + w * dense`, with `w` chosen per tier on
   validation queries.
3. **Normalize per tier.**
   - Full tier: `rescore`, min-max on both sides, `w = 0.65`.
   - Lite tier: `rescore-max`, BM25 divided by the query's best BM25 score and the raw cosine,
     `w = 0.45`.
4. Keep RRF and the unrescored `score` rule available behind `--fusion` and `Fusion`, for
   comparison and for embedders whose scores cannot be mixed.

## Why the tiers normalize differently

Min-max maps the weakest candidate of each list to 0 whatever its absolute score. For BM25 this
throws away information: in a list of strong matches, the last one is still a strong match.
Dividing by the best score keeps BM25 proportional.

Raw cosines can only be used directly when they spread across their range. The static model's
do; bge-small's sit in a narrow band where unrelated texts still score high, so used raw they
barely separate candidates and must be stretched. The measurements agree: max fusion improved the
lite tier and lost to min-max for the full tier at every weight tried.

## Evidence

**Rescore against RRF**, lite tier, BEIR test, nDCG@10, before the analyzer change (`bench/results/baseline/`):

| | BM25 | Dense | RRF | Rescore 0.4 |
|---|---|---|---|---|
| SciFact | 0.661 | 0.622 | 0.664 | **0.693** |
| NFCorpus | 0.309 | 0.302 | 0.324 | **0.330** |
| FiQA | 0.237 | 0.186 | 0.233 | **0.251** |
| ArguAna | 0.405 | 0.450 | 0.462 | **0.470** |

**Full tier**, test, nDCG@10, difference against RRF in parentheses (`bench/results/2026-10-26-full-tier-v2/`):

| | RRF | Rescore 0.65 |
|---|---|---|
| SciFact | 0.7266 | **0.7424** (+0.016) |
| NFCorpus | **0.3634** | 0.3622 (-0.001) |
| ArguAna | 0.5512 | **0.6062** (+0.055) |
| Open RAG Bench | 0.7867 | **0.8268** (+0.040) |

The full tier's weight: validation mean nDCG@10 over four sets was 0.6345 at 0.55, 0.6374 at 0.6,
**0.6392 at 0.65**, 0.6379 at 0.7, and 0.6351 at 0.8. On test, 0.65 beat the previous 0.3
significantly on all four sets, by 0.014 to 0.051.

**Max fusion for the lite tier**, test, nDCG@10 (`bench/results/2026-10-26-quality-sweep/`):

| | rescore 0.4 | rescore-max 0.45 |
|---|---|---|
| Tuning mean (SciFact, NFCorpus, ArguAna, FiQA, ORB) | 0.4480 | **0.4524** |
| Held-out mean (TREC-COVID, SciDocs, Quora, Touché) | 0.4922 | **0.4993** |
| TREC-COVID | 0.6441 | **0.6797** (significant) |
| Touché-2020 | **0.3284** | 0.3183 (not significant) |

## Alternatives considered

| Alternative | Outcome |
|---|---|
| RRF (the common default) | Below rescore on 3 of 4 full-tier sets and every lite set; below BM25 alone with a weak dense leg |
| `score` fusion without rescoring | Penalizes one-sided candidates; kept only for comparison |
| A per-query weight, from query length or leg overlap | Chosen per bucket on validation, every bucket picked the same weight ([Planner](../concepts/planner.md)) |
| Dense pseudo-relevance feedback (fold the top fused vectors into the query, search again) | Held-out +0.001 to +0.002 over max fusion, query time up to 2x on large sets; dropped |
| Always rerank with a cross-encoder | About 180 ms for the top 8 on CPU; +0.08 on FiQA, -0.06 to -0.09 on Open RAG Bench; off by default |

## Consequences

- Each query computes about 100 extra int8 dot products, 0.2 to 0.4 ms, and needs the vector
  file mapped.
- The weights belong to a model and an analyzer. Changing either means re-choosing them on
  validation queries, which the planner's trace and replay make cheap.
- Scores are not comparable across queries or tiers. The API documents this; callers must not
  threshold on a fixed score.
- Gyrfalcon's defaults are tuned while other frameworks' are not. The benchmark pages disclose
  this and report held-out sets ([Methodology](../benchmarks/methodology.md#disclosures)).

---

<sub>Next: [0002. Two named tiers](0002-tier-system.md)</sub>
