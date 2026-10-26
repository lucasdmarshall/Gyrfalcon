<sub>[Docs](../README.md) / Design records</sub>

# 0003. A query term repeated n times weighs n in BM25

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-10-26 |
| Code | `crates/gyrfalcon/src/lexical/mod.rs` (`query_terms`), `buffer.rs`, `lexical/reference.rs` |
| Evidence | `bench/results/2026-10-26-quality-sweep/` |

## Context

BM25 sums a score per query term. Gyrfalcon originally summed over the query's distinct terms, so
a query that said "tax" five times scored like one that said it once.

Lucene and Tantivy do not behave that way. A query parsed into a boolean query of term clauses
gets one clause per occurrence, and each clause adds its score, so a term repeated n times
contributes n times. Gyrfalcon's BM25 is tested to match Tantivy's scores exactly, and that test
had been deduplicating query terms to make the comparison pass.

Short keyword queries rarely repeat a term, so the difference is invisible on most benchmarks. It
matters for long queries: ArguAna's queries are whole arguments, often over 100 words, where the
repeated words are usually the topic.

## Decision

Score each distinct query term once and multiply its contribution by its count in the analyzed
query:

```text
BM25(q, d) = sum over distinct terms t of q:  count(t, q) * idf(t) * tf_part(t, d)
```

This gives the same score as one clause per occurrence, with one postings traversal per distinct
term. It applies identically to sealed segments and to the write buffer, so a chunk scores the
same before and after it is sealed. Counting happens after analysis, so "Taxes" and "tax" count
as the same term.

The Tantivy differential test now keeps duplicate clauses, and accepts float ties at the cut-off,
where the two engines may order equal scores differently.

## Evidence

Test splits, nDCG@10 (`bench/results/2026-10-26-quality-sweep/`):

| | Before | After |
|---|---|---|
| ArguAna, BM25 alone | 0.403 | **0.482** |
| ArguAna, lite fused | | +0.014 |
| ArguAna, full fused | about 0.602 | **0.612** |
| Other sets, lite fused | | within ±0.003 |
| SciFact, full fused | 0.7431 | 0.7430 |
| NFCorpus, full fused | 0.3620 | 0.3608 |

Query latency and index time did not change: the number of postings read is the same.

## Alternatives considered

| Alternative | Outcome |
|---|---|
| Keep distinct terms only | Disagrees with Lucene and Tantivy; loses 0.08 BM25 nDCG@10 on ArguAna |
| Tune BM25's `k1` and `b` instead | Results were dataset-dependent: `b = 0.4` raised Touché BM25 from 0.333 to 0.447 but dropped ArguAna from 0.403 to 0.316. Defaults kept |

A saturating query-term weight, such as Okapi BM25's `k3` component, was not evaluated. It would
damp very long queries and is a candidate if a corpus with long, repetitive queries shows the
linear count overweighting a term.

## Consequences

- Scores now agree with Lucene and Tantivy for repeated terms, so results can be cross-checked
  against those engines without special cases.
- Long queries put more weight on the words they repeat. That is the intended effect, and on the
  one long-query benchmark it is a large gain.
- No change to the on-disk format; existing collections score this way as soon as the engine is
  updated.

---

<sub>Previous: [0002. Two named tiers](0002-tier-system.md)</sub>
