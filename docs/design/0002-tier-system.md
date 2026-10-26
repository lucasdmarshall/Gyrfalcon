<sub>[Docs](../README.md) / Design records</sub>

# 0002. Two named tiers instead of free model configuration

| | |
|---|---|
| Status | Accepted |
| Decided | 2026-10-24 (tiers); 2026-10-26 (moved into the library as `gyrfalcon::Tier`) |
| Code | `crates/gyrfalcon/src/tier.rs`, `manifest.rs`, `index.rs` (`open_counting`) |
| Evidence | `bench/results/2026-10-26-lite-refresh/`, `2026-10-26-cloud-4core-bge-h2h/`, spec "Lite model candidates" |

## Context

The first version had one default: bge-small-en-v1.5 through ONNX Runtime. It gave good quality
but made every user pay for a transformer at ingest, about 25 documents per second on 4 cores,
and required shipping a native runtime. Many RAG corpora are large and change often, and many
machines that run them are small.

A static Model2Vec model (potion-retrieval-32M) indexes 500 to 600 times faster with no runtime,
and loses 0.02 to 0.12 nDCG@10. Neither model is right for everyone.

Several settings must agree for a collection to work: the model, its vector width, the stored
number format, the analyzer, and the fusion rule and weight tuned for that model. Exposing them as
independent options invites combinations that were never measured, and one mismatch (querying
with a different model than the one that built the index) returns plausible but wrong results
with no error.

## Decision

1. Offer two named tiers, `lite` and `full`. Each fixes the model and its files, the model id,
   the vector width, the embedder, and the fusion rule with its weight.
2. Define them once, in `gyrfalcon::Tier`, and open collections through it from Python, the
   command line, and Rust, so a collection built by one opens in the others.
3. Record the identity (format version, model id, dimension, quantization, analyzer) in the
   manifest at creation, and compare it on every open, before mapping any file or loading any
   model. Refuse a mismatch with the field and both values.
4. Make `lite` the default.
5. Keep expert overrides (another embedder, fusion rule, or weight) in the command line and the
   Rust API, under the same identity check. Keep the Python API to the two tiers.

## Why `lite` is the default

A default should be the choice that is never badly wrong. `lite` runs everywhere, indexes in
seconds, needs nothing installed, and scores within 0.02 to 0.04 of `full` on SciFact and
NFCorpus. ArguAna, 0.12 apart, is the exception. `full` is a deliberate step up for
corpora where ingest time is affordable. A user who starts with `lite` and needs more changes one
argument and rebuilds; a user who starts with `full` on a small machine may conclude the library
is slow.

## Evidence

4-vCPU VM, BEIR test (`2026-10-26-lite-refresh`, `2026-10-26-cloud-4core-bge-h2h`, `2026-10-26-quality-sweep`):

| | lite | full |
|---|---|---|
| SciFact nDCG@10 | 0.709 | 0.743 |
| NFCorpus nDCG@10 | 0.344 | 0.361 |
| ArguAna nDCG@10 | 0.491 | 0.612 |
| SciFact index time | 0.32 s | 193 s |
| SciFact query p50 | 1.7 ms | 9.1 ms |
| SciFact peak memory | 268 MB | 499 MB |

Another static candidate, `static-retrieval-mrl-en-v1`, scored below potion on three of four BEIR
sets with the default fusion, and its dense leg alone was weaker on SciFact (0.598 against 0.632).
Potion stayed as the lite model.

## Alternatives considered

| Alternative | Why not |
|---|---|
| One model for everyone (bge-small) | Slow ingest on small machines, native runtime always required |
| One static model for everyone | Leaves 0.02 to 0.12 nDCG@10 on the table for users who can afford more |
| Free configuration of model, width, format, and fusion | Unmeasured combinations, silent mismatches, more to document and support |
| Choosing the tier automatically from the corpus | No signal available at creation predicts which tier wins on a user's queries |
| A third, larger tier (for example bge-base) | Not measured yet; would follow the same pattern if added |

## Consequences

- A collection cannot change tier in place. Moving to `full` is a rebuild from the sources.
- Each tier's fusion weight is tuned and documented for its model. A new tier needs its own
  tuning on validation queries and a held-out check.
- Opening with the wrong tier fails fast and says why, before any expensive work.
- Users who need another model use the command line or Rust with the same guarantees, or
  serve the collection over HTTP for other languages.

---

<sub>Previous: [0001. Hybrid fusion](0001-hybrid-fusion.md) · Next: [0003. Query term frequency](0003-query-term-frequency.md)</sub>
