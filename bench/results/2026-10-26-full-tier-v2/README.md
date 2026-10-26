# Full tier fusion weight with analyzer v2, 2026-10-26

`gyr bench trace --tier full` (int8 bge-small-en-v1.5, `gyr-text-v2`) over SciFact, NFCorpus,
ArguAna, and the Open RAG Bench text-only fixture, then `gyr bench replay`. Weights were chosen on
the validation splits and confirmed once on test; intervals are paired bootstrap 95% against the
reference policy (* = significant). Cloud VM, 4 vCPU; replayed latencies are not comparable
across runs made while other jobs ran.

Validation, nDCG@10 (difference vs rescore 0.3):

| policy | SciFact | NFCorpus | ArguAna | ORB |
|---|---|---|---|---|
| lexical | 0.6989 (-0.0252*) | 0.2908 (-0.0339*) | 0.3866 (-0.1601*) | 0.7992 (-0.0341*) |
| dense | 0.7456 (+0.0215*) | 0.3239 (-0.0008) | 0.5950 (+0.0482*) | 0.7056 (-0.1277*) |
| rrf | 0.7488 (+0.0247*) | 0.3399 (+0.0152*) | 0.5355 (-0.0112) | 0.8154 (-0.0179*) |
| rescore 0.3 (old default) | 0.7241 | 0.3247 | 0.5467 | 0.8333 |
| rescore 0.5 | 0.7463 (+0.0223*) | 0.3359 (+0.0112*) | 0.5881 (+0.0413*) | 0.8522 (+0.0189*) |
| rescore 0.6 | 0.7587 (+0.0347*) | 0.3403 (+0.0157*) | 0.5954 (+0.0487*) | 0.8551 (+0.0218*) |
| rescore 0.7 | 0.7660 (+0.0419*) | 0.3409 (+0.0162*) | 0.5963 (+0.0496*) | 0.8485 (+0.0152*) |

Mean over the four by weight: 0.55 0.6345, 0.6 0.6374, **0.65 0.6392**, 0.7 0.6379, 0.75 0.6375,
0.8 0.6351, 0.9 0.6188 (`replay-validation-fine.json`). 0.65 is no dataset's significant loss
against 0.6.

Test, nDCG@10 (difference vs rescore 0.3):

| policy | SciFact | NFCorpus | ArguAna | ORB |
|---|---|---|---|---|
| rrf | 0.7266 (+0.0116) | 0.3634 (+0.0151*) | 0.5512 (-0.0042) | 0.7867 (-0.0256*) |
| rescore 0.3 | 0.7150 | 0.3482 | 0.5554 | 0.8124 |
| rescore 0.4 | 0.7233 (+0.0083*) | 0.3524 (+0.0042*) | 0.5819 (+0.0265*) | 0.8208 (+0.0084*) |
| **rescore 0.65** | **0.7424 (+0.0274*)** | **0.3622 (+0.0139*)** | **0.6062 (+0.0508*)** | **0.8268 (+0.0144*)** |

The full tier's default dense weight is now 0.65 (`FULL_RESCORE_DENSE_WEIGHT`). With v2's
stemming the lexical leg got stronger, but bge-small's dense leg is stronger still, so the
weight moved toward it; the lite tier's static embeddings keep 0.4.
