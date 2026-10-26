# v2 beta: parked features

Designs for Gyrfalcon v2 that are written down but **not implemented and not measured**. Each is
parked because it changes what a collection is, costs more than the v1 speed budget allows, or
carries a risk of fitting the benchmarks rather than the user's data. Nothing here ships in v1,
and nothing here may change a v1 default.

A feature leaves this folder only through its own staged plan: every stage measured on the
held-out protocol below, with its gates met, before the next stage starts.

| feature | document | status |
|---|---|---|
| Corpus distillation: a static embedding table distilled from bge-small for each collection | [corpus-distillation.md](corpus-distillation.md) | design only; parked 2026-10-26 |

## Protocol every v2 feature is measured on

Fixed on 2026-10-26 (`bench/results/2026-10-26-quality-sweep`):

- **Tuning sets** for choices: SciFact, NFCorpus, ArguAna, FiQA, and Open RAG Bench, using their
  validation splits where they exist (SciFact `train`, NFCorpus and FiQA `dev`, the others a
  hash split of `test`).
- **Report** on the tuning sets' test splits and on held-out sets never used for a choice:
  TREC-COVID, SciDocs, Quora, Touché-2020. A feature that needs a new choice may add held-out
  sets, never remove one.
- **Significance:** paired bootstrap, 95% interval, per dataset.
- **Speed and memory** next to quality on the same run: query p50 and p99, index seconds, peak
  RSS, on the 4-core cloud VM and, for ingest, one pinned core.

v1 lite baseline on that protocol (commit 90f1580), nDCG@10:

| scifact | nfcorpus | arguana | fiqa | trec-covid | scidocs | quora | touche | tuning | held-out |
|---|---|---|---|---|---|---|---|---|---|
| 0.7088 | 0.3435 | 0.4912 | 0.2659 | 0.6797 | 0.1646 | 0.8344 | 0.3183 | 0.4524 | 0.4993 |
