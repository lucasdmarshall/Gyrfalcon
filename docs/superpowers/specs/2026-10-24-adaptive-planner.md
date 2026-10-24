# Gyrfalcon Adaptive Retrieval Planner: spec and evaluation protocol

Date: 2026-10-24
Status: draft. No planner code is written until the baseline in step P0 is locked.

## Thesis

Gyrfalcon's advantage is not that each component is the fastest. It is that, for each query,
it spends compute only where that compute is likely to change the answer, and it can show on
public benchmarks that the quality it gives up is within a stated bound.

This is a hypothesis. Nothing in this document is a measured result until the protocol below
produces it.

## Scope

In scope: deciding per query whether to embed, whether and how deeply to rerank, how to
weight the dense leg, and which evidence to return inside a context budget.

Out of scope: a new inference runtime, a new ANN index, a new storage engine. The planner
uses the components that exist: Tantivy BM25 with collection statistics, USearch, rescore
fusion, the static and ort embedders, and the MiniLM cross-encoder.

## API

`retrieve(query, RetrieveOptions)` keeps its meaning. A new call carries the budget:

```
plan_retrieve(query, Plan) -> PlannedHits

Plan {
  limit: usize,
  rerank: Never | Always | Auto,       // Auto lets the planner decide per query
  latency_budget_ms: Option<f32>,      // soft target for this call
  context_tokens: Option<usize>,       // evidence budget for the returned bodies
}

PlannedHits {
  hits, chunk_ids,
  trace: PlanTrace,                    // which steps ran, their timings, and the signals
}
```

`PlanTrace` is always returned so a caller and the benchmark can see why a step was skipped.

## Signals

Computed after both legs return, before any rerank. Every signal is cheap (no model call).

| Signal | Definition |
|---|---|
| `margin` | fused score of rank 1 minus rank 2, after min-max normalization |
| `agreement` | rank correlation (Kendall tau) between the BM25 and dense orders over their shared top 10 |
| `overlap` | Jaccard of the BM25 top 10 and dense top 10 id sets |
| `top_agree` | the BM25 rank-1 and dense rank-1 ids are the same |
| `lex_peak` | top BM25 score over the mean of the BM25 top 10 |
| `dense_peak` | top cosine minus the mean cosine of the dense top 10 |
| `idf_max` | highest query-term IDF from the collection statistics |
| `identifier` | the query has a token with a digit, an underscore, `::`, or mixed case inside a word |
| `query_tokens` | number of `gyr-text-v1` tokens |
| `budget` | the caller's latency budget and context budget |

No signal is assumed useful. Each must show value on validation data (see P3 and P4).

## Decisions the planner makes

1. Rerank: none, or rerank the top `k` in {8, 16, 32} with bodies truncated to {128, 256}
   tokens.
2. Dense weight `w` for rescore fusion, chosen per query from a small set
   {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7}.
3. Skip the dense leg entirely when a rule says BM25 alone is enough (saves the embed call
   on the full tier; the lite embed is already ~30 microseconds).
4. Evidence selection under `context_tokens` (P5).

The policy is a small, deterministic function: threshold rules or a logistic model whose
coefficients ship as JSON. No ML runtime is added.

## Evaluation protocol

### Datasets

| Dataset | Validation | Test |
|---|---|---|
| BEIR SciFact | `train` qrels | `test` qrels |
| BEIR NFCorpus | `dev` qrels | `test` qrels |
| BEIR FiQA-2018 | `dev` qrels | `test` qrels |
| BEIR ArguAna | 50% of `test` queries by hash | the other 50% |
| Open RAG Bench text-only | 50% of the 1914 questions by hash | the other 50% |

Hash split: `sha256(query_id)` first byte even goes to validation. The split is fixed in code
and in this document before any planner experiment runs.

Every threshold, weight, and coefficient is chosen on validation only. Test numbers are
reported once per planner version and never used to choose anything. A planner version that
was tuned after seeing its test numbers is discarded and the next version uses a fresh rule.

### Trace-and-replay

Running every policy end to end is slow on the full tier. Instead, each validation and test
query runs once in "trace" mode, which computes everything any policy could use:

- BM25 top 100 with scores, dense top 100 with cosines, exact cosines for their union.
- Cross-encoder scores for the fused top 32 at 128 and at 256 tokens.
- Wall time of each step, measured single-flight on the bench machine.

Traces are JSONL under `bench/traces/`. A policy is evaluated by replaying traces: its quality
is computed from the stored scores, and its latency is the sum of the measured times of the
steps it chose. A final check runs the chosen policy live and must match the replay within
1% on nDCG@10 and 10% on p50 latency; otherwise the replay is not trusted.

### Baselines (locked in P0)

| Id | Policy |
|---|---|
| B-lex | BM25 only |
| B-dense | dense only |
| B-rrf | RRF, weights 1 and 1 |
| B-fixed | rescore with the tier's fixed `w` (lite 0.2, full 0.3), no rerank |
| B-best-w | rescore with the single `w` best on validation, no rerank |
| B-rerank | B-best-w, then always rerank top 32 at 256 tokens (the quality ceiling) |

### Metrics

Per dataset, on test: nDCG@10, Recall@10, Recall@100, MRR@10; latency p50, p95, p99
(single-flight and in-flight 4); rerank invocation rate; cross-encoder pairs per query;
embed calls per query. Latency is measured pinned to 1, 2, and 4 cores.

Quality differences carry a 95% confidence interval from a paired bootstrap over queries
(10,000 resamples, seed 1). A difference whose interval contains 0 is reported as "no
significant difference", not as a win.

### Success criteria

Adaptive rerank (P3) succeeds on a dataset when, against B-best-w and B-rerank:

- it keeps at least 90% of B-rerank's nDCG@10 gain over B-best-w, and
- it reranks at most 50% of queries, and
- its nDCG@10 is not significantly below B-best-w.

If B-rerank is not significantly better than B-best-w on a dataset, adaptive rerank has
nothing to save there and the planner must choose "no rerank" for at least 90% of its
queries.

Query-adaptive fusion (P4) succeeds when its nDCG@10 is significantly above B-best-w on at
least two datasets and not significantly below it on any.

The planner ships as the default only if it meets its criteria on a majority of datasets and
is never significantly below B-fixed.

## Embedding cache (P2)

LRU keyed by the hash of `prepared.embed_text`, bounded in bytes (default 8 MiB). Reported:
hit rate, bytes used, evictions, and the latency it saves. Benchmark queries are unique, so
the hit rate is measured on a replayed query log with Zipf(1.1) repetition over the test
queries, and stated as synthetic. On the lite tier the cache must not cost more than it
saves; if the embed (~30 microseconds) is cheaper than a lookup plus the memory, the lite
tier ships with the cache off.

## Context-budgeted evidence (P5)

Given `context_tokens`, remove near-duplicates (same document and overlapping byte spans, or
token-set Jaccard >= 0.8), then pick hits greedily by score per token until the budget is
full. Measured on Open RAG Bench with the product chunker (which creates overlaps): recall of
the relevant section within budgets of 512, 1024, and 2048 tokens, against plain top-k
truncation.

## Steps

| Step | Deliverable | Gate |
|---|---|---|
| P0 | Baseline reports for both tiers on all datasets, stored in `bench/results/baseline/` | none |
| P1 | Trace mode, the validation/test split, replay, bootstrap CIs | replay matches live runs |
| P2 | `Plan`, explicit budgets, `PlanTrace`, embedding cache | no quality change vs B-fixed |
| P3 | Adaptive rerank policy chosen on validation | success criteria above |
| P4 | Query-adaptive dense weight | success criteria above |
| P5 | Context-budgeted dedupe | beats top-k truncation at equal budget |

Each step's test numbers go into `bench/results/` with the command that produced them.
